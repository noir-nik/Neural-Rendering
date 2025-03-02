#include "../Common/Vulkan/CheckResult.h"
#include "Shaders/SDFConstants.h"
import std;
#include "Shaders/SDFWeights.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;

using namespace ng::Types;
using namespace ng::Utils;

bool bVerbose    = false;
bool bSlang      = true;
bool bValidation = true;

static constexpr u32 kApiVersion = vk::ApiVersion13;

// std::size_t const kSdfWeightsSize = (sizeof(kSDFWeights0) + sizeof(kSDFBias0) +
// 									 sizeof(kSDFWeights1) + sizeof(kSDFBias1) +
// 									 sizeof(kSDFWeights2) + sizeof(kSDFBias2) +
// 									 sizeof(kSDFWeights3) + sizeof(kSDFBias3));

static constexpr char const* kEnabledLayers[]           = {"VK_LAYER_KHRONOS_validation"};
static constexpr char const* kEnabledDeviceExtensions[] = {
	vk::KHRSwapchainExtensionName,
	vk::NVCooperativeVectorExtensionName,
	vk::NVCooperativeVectorExtensionName,
	vk::EXTShaderReplicatedCompositesExtensionName,
};

struct PhysicalDevice : public VulkanRHI::PhysicalDevice {
	PhysicalDevice() {
		AddToPNext(GetFeatures2(), cooperative_vector_features);
		AddToPNext(GetFeatures2(), shader_replicated_composites_features);
		AddToPNext(GetProperties2(), cooperative_vector_properties);
	}
	bool IsSuitable(vk::SurfaceKHR const& surface) {
		bool const bSupportsExtensions        = SupportsExtensions(kEnabledDeviceExtensions);
		bool const bSupportsQueues            = SupportsQueue({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
		bool const bSupportsCooperativeVector = cooperative_vector_features.cooperativeVector == vk::True;
		bool const bSupportsShaderReplicated  = shader_replicated_composites_features.shaderReplicatedComposites == vk::True;
		if (bSupportsExtensions && bSupportsQueues && bSupportsCooperativeVector && bSupportsShaderReplicated) {
			return true;
		}
		return false;
	}

	vk::PhysicalDeviceCooperativeVectorFeaturesNV           cooperative_vector_features{};
	vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT shader_replicated_composites_features{};
	vk::PhysicalDeviceCooperativeVectorPropertiesNV         cooperative_vector_properties{};
	u32                                                     graphics_queue_family_index = std::numeric_limits<u32>::max();
};

struct Vertex {
	struct {
		float x, y, z;
	} position;
};

constexpr Vertex kVertices[] = {
	Vertex{-1.0, -1.0, 0.0},
	Vertex{-1.0, 1.0, 0.0},
	Vertex{1.0, -1.0, 0.0},
	Vertex{-1.0, 1.0, 0.0},
	Vertex{1.0, 1.0, 0.0},
	Vertex{1.0, -1.0, 0.0},
};

class SDFSample {
public:
	vk::ComponentTypeKHR kSrcComponentType = SRC_COMPONENT_TYPE;
	vk::ComponentTypeKHR kDstMatrixType    = DST_MATRIX_TYPE;
	vk::ComponentTypeKHR kDstVectorType    = DST_VECTOR_TYPE;
	vk::CooperativeVectorMatrixLayoutNV kDstLayout = MATRIX_LAYOUT;

	~SDFSample();

	void Init();
	void Run();
	void Destroy();
	void CreateInstance();
	void SelectPhysicalDevice();
	void GetPhysicalDeviceInfo();
	void CreateDevice();
	void CreateVmaAllocator();

	void CreateDescriptorSetLayout();
	void CreateDescriptorPool();
	void CreateDescriptorSet();

	void CreateSwapchain();

	void CreatePipelineLayout();
	void CreatePipeline();

	// void BuildNetwork();
	void CreateBuffers(std::size_t size);
	void UploadNetworkWeights();
	void UploadNetworkRaw();

	void RecordCommands();
	void DrawWindow();
	void RecreateSwapchain(int width, int height);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }

	Window window{};
	struct {
		float x = 300.0f;
		float y = 300.0f;
	} mouse;

	vk::Instance                    instance{};
	vk::AllocationCallbacks const*  allocator{nullptr};
	VmaAllocator                    vma_allocator{};
	vk::PipelineCache               pipeline_cache{nullptr};
	vk::DebugUtilsMessengerEXT      debug_messenger{};
	std::span<char const* const>    enabled_layers{};
	std::vector<vk::PhysicalDevice> vulkan_physical_devices{};
	PhysicalDevice                  physical_device{};
	vk::Device                      device{};
	VulkanRHI::Swapchain            swapchain{};
	bool                            swapchain_dirty = false;
	vk::SurfaceKHR                  surface{};

	vk::Queue queue{};
	u32       queue_family_index = ~0u;

	vk::DescriptorSetLayout descriptor_set_layout{};
	vk::DescriptorPool      descriptor_pool{};
	vk::DescriptorSet       descriptor_set{};

	vk::PipelineLayout pipeline_layout{};
	vk::Pipeline       pipeline{};

	VulkanRHI::Buffer staging_buffer{};
	VulkanRHI::Buffer vertex_buffer{};
	VulkanRHI::Buffer sdf_weights_buffer{};

	ng::VulkanCoopVecNetwork network = {
		ng::Linear(3, 16),
		ng::Linear(16, 16),
		ng::Linear(16, 16),
		ng::Linear(16, 1),
	};
	// std::vector<std::byte> network_parameters;
};

static SDFSample* gSDFSample = nullptr;
static void       FramebufferSizeCallback(Window* window, int width, int height) {
    gSDFSample->swapchain_dirty = true;
    if (width <= 0 || height <= 0) return;
    gSDFSample->RecreateSwapchain(width, height);
}

static void WindowRefreshCallback(Window* window) {
	int x, y, width, height;
	window->GetRect(x, y, width, height);
	if (width <= 0 || height <= 0) return;
	gSDFSample->DrawWindow();
}

static void CursorPosCallback(Window* window, double xpos, double ypos) {
	gSDFSample->mouse.x = static_cast<float>(xpos);
	gSDFSample->mouse.y = static_cast<float>(ypos);
}

static void KeyCallback(Window* window, int key, int scancode, int action, int mods) {
	switch (key) {
	case 256:
		window->SetShouldClose(true);
		break;
	}
}

void SDFSample::Init() {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();
	window.Init({.x = 30, .y = 30, .width = 800, .height = 600, .title = "SDF"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	window.GetWindowCallbacks().windowRefreshCallback   = WindowRefreshCallback;
	window.GetInputCallbacks().cursorPosCallback        = CursorPosCallback;
	window.GetInputCallbacks().keyCallback              = KeyCallback;
	CreateInstance();

	CHECK_RESULT(WindowManager::CreateWindowSurface(instance, reinterpret_cast<GLFWwindow*>(window.GetHandle()), GetAllocator(), &surface));

	SelectPhysicalDevice();
	GetPhysicalDeviceInfo();

	CreateDevice();
	CreateVmaAllocator();

	CreateDescriptorSetLayout();
	CreateDescriptorPool();
	CreateDescriptorSet();

	CreateSwapchain();

	CreatePipelineLayout();
	CreatePipeline();

	LoadInstanceCooperativeVectorFunctionsNV(instance);

	// BuildNetwork();
	CHECK_RESULT(network.UpdateOffsetsAndSize(device,
											  kDstLayout,
											  kDstMatrixType,
											  kDstVectorType));

	std::printf("Parameters: %zu\n", network.GetParametersSize());
	for (auto& layer : network.GetLayers()) {
		ng::Linear& linear = layer.Get<ng::Linear>();
		std::printf("Weights: %zu, Biases: %zu\n", linear.GetWeightsOffset(), linear.GetBiasesOffset());
	}
	// network_parameters.resize(network.GetParametersSize());
	CreateBuffers(network.GetParametersSize());
	UploadNetworkWeights();

	auto [result, cooperative_vector_properties] = physical_device.getCooperativeVectorPropertiesNV();
	CHECK_RESULT(result);

	if (bVerbose) {
		std::printf("=== VkCooperativeVectorPropertiesNV ===\n");
		for (auto& property : cooperative_vector_properties) {
			std::printf("inputType: %-8s ", vk::to_string(property.inputType).c_str());
			std::printf("inputInterpretation: %-13s ", vk::to_string(property.inputInterpretation).c_str());
			std::printf("matrixInterpretation: %-13s ", vk::to_string(property.matrixInterpretation).c_str());
			std::printf("biasInterpretation: %-13s ", vk::to_string(property.biasInterpretation).c_str());
			std::printf("resultType: %-10s ", vk::to_string(property.resultType).c_str());
			std::printf("transpose: %-10u ", property.transpose);
			std::printf("\n");
		}
	}
}

SDFSample::~SDFSample() { Destroy(); }

void SDFSample::Destroy() {

	if (device) {
		CHECK_RESULT(device.waitIdle());

		staging_buffer.Destroy();
		vertex_buffer.Destroy();
		sdf_weights_buffer.Destroy();

		device.destroyPipeline(pipeline, GetAllocator());
		device.destroyPipelineLayout(pipeline_layout, GetAllocator());

		device.destroyDescriptorSetLayout(descriptor_set_layout, GetAllocator());
		device.destroyDescriptorPool(descriptor_pool, GetAllocator());

		swapchain.Destroy();
		device.destroyPipelineCache(pipeline_cache, GetAllocator());

		vmaDestroyAllocator(vma_allocator);

		device.destroy(GetAllocator());
		device = vk::Device{};
	}

	if (instance) {
		instance.destroySurfaceKHR(surface, GetAllocator());
		if (bValidation) {
			instance.destroyDebugUtilsMessengerEXT(debug_messenger, GetAllocator());
		}
		instance.destroy(GetAllocator());
		instance = vk::Instance{};
	}
	window.Destroy();
	WindowManager::Terminate();
}

void SDFSample::CreateInstance() {
	vk::ApplicationInfo applicationInfo{.apiVersion = kApiVersion};

	u32 glfw_extensions_count;

	char const** glfw_extensions = WindowManager::GetRequiredInstanceExtensions(&glfw_extensions_count);

	std::vector<char const*> enabledExtensions(glfw_extensions, glfw_extensions + glfw_extensions_count);
	if (bValidation) {
		enabled_layers = kEnabledLayers;
		enabledExtensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	vk::DebugUtilsMessengerCreateInfoEXT constexpr kDebugUtilsCreateInfo = {
		.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
						   //    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
						   //    vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
						   vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
					   vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
					   vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
					   vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding,
		.pfnUserCallback = DebugUtilsCallback,
		.pUserData       = nullptr,
	};

	vk::InstanceCreateInfo info{
		.pNext                   = bValidation ? &kDebugUtilsCreateInfo : nullptr,
		.pApplicationInfo        = &applicationInfo,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(enabledExtensions)),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};
	CHECK_RESULT(vk::createInstance(&info, GetAllocator(), &instance));

	if (bValidation) {
		LoadInstanceDebugUtilsFunctionsEXT(instance);
		CHECK_RESULT(instance.createDebugUtilsMessengerEXT(&kDebugUtilsCreateInfo, allocator, &debug_messenger));
	}

	vk::Result result;
	std::tie(result, vulkan_physical_devices) = instance.enumeratePhysicalDevices();
	CHECK_RESULT(result);
}

void SDFSample::SelectPhysicalDevice() {
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface)) {
			if (bVerbose) {
				auto const& properties = physical_device.cooperative_vector_properties;
				std::printf("=== VkPhysicalDeviceCooperativeVectorPropertiesNV ===\n");
				std::printf("cooperativeVectorSupportedStages: %s\n", vk::to_string(properties.cooperativeVectorSupportedStages).c_str());
				std::printf("cooperativeVectorTrainingFloat16Accumulation: %s\n", FormatBool(properties.cooperativeVectorTrainingFloat16Accumulation));
				std::printf("cooperativeVectorTrainingFloat32Accumulation: %s\n", FormatBool(properties.cooperativeVectorTrainingFloat32Accumulation));
				std::printf("maxCooperativeVectorComponents: %s\n", FormatBool(properties.maxCooperativeVectorComponents));
			}
			return;
		}
	}

	std::printf("No suitable physical device found\n");
	// std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	std::getchar();
	std::exit(1);
}

void SDFSample::GetPhysicalDeviceInfo() {
}

void SDFSample::CreateDevice() {
	float const queue_priorities[] = {1.0f};

	auto [result, index] = physical_device.GetQueueFamilyIndex({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
	if (result != vk::Result::eSuccess || !index.has_value()) {
		std::printf("Failed to get graphics queue family index with surface support\n");
		CHECK_RESULT(result);
	}

	queue_family_index = index.value();

	vk::DeviceQueueCreateInfo queue_create_infos[] = {
		{
			.queueFamilyIndex = queue_family_index,
			.queueCount       = 1,
			.pQueuePriorities = queue_priorities,
		}};

	vk::StructureChain features{
		vk::PhysicalDeviceFeatures2{},
		vk::PhysicalDeviceVulkan11Features{.storageBuffer16BitAccess = vk::True},
		vk::PhysicalDeviceVulkan12Features{.shaderFloat16 = vk::True, .bufferDeviceAddress = vk::True},
		vk::PhysicalDeviceVulkan13Features{.synchronization2 = vk::True, .dynamicRendering = vk::True},
		vk::PhysicalDeviceCooperativeVectorFeaturesNV{.cooperativeVector = vk::True, .cooperativeVectorTraining = vk::True},
		vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT{.shaderReplicatedComposites = vk::True},
	};

	vk::DeviceCreateInfo info{
		.pNext                   = &features.get<vk::PhysicalDeviceFeatures2>(),
		.queueCreateInfoCount    = static_cast<u32>(std::size(queue_create_infos)),
		.pQueueCreateInfos       = queue_create_infos,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(kEnabledDeviceExtensions)),
		.ppEnabledExtensionNames = kEnabledDeviceExtensions,
	};

	CHECK_RESULT(physical_device.createDevice(&info, GetAllocator(), &device));
	queue = device.getQueue(queue_create_infos[0].queueFamilyIndex, 0);
}

void SDFSample::CreateVmaAllocator() {
	VmaVulkanFunctions vulkan_functions = {
		.vkGetInstanceProcAddr = &vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr,
	};

	using ::VmaAllocatorCreateFlagBits;
	VmaAllocatorCreateInfo info = {
		.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT |
				 VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT |
				 VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice   = physical_device,
		.device           = device,
		.pVulkanFunctions = &vulkan_functions,
		.instance         = instance,
		.vulkanApiVersion = kApiVersion,
	};
	CHECK_RESULT(vk::Result(vmaCreateAllocator(&info, &vma_allocator)));
}

void SDFSample::CreateDescriptorSetLayout() {
	vk::DescriptorSetLayoutBinding descriptor_set_layout_binding{
		.binding         = 0,
		.descriptorType  = vk::DescriptorType::eStorageBuffer,
		.descriptorCount = 1,
		.stageFlags      = vk::ShaderStageFlagBits::eFragment,
	};

	vk::DescriptorSetLayoutCreateInfo info{
		.flags        = {},
		.bindingCount = 1,
		.pBindings    = &descriptor_set_layout_binding,
	};

	CHECK_RESULT(device.createDescriptorSetLayout(&info, GetAllocator(), &descriptor_set_layout));
}

void SDFSample::CreateDescriptorPool() {
	vk::DescriptorPoolSize descriptor_pool_sizes[] = {
		{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = 1},
	};

	vk::DescriptorPoolCreateInfo info{
		.flags         = {},
		.maxSets       = 1,
		.poolSizeCount = 1,
		.pPoolSizes    = descriptor_pool_sizes,
	};

	CHECK_RESULT(device.createDescriptorPool(&info, GetAllocator(), &descriptor_pool));
}

void SDFSample::CreateDescriptorSet() {
	vk::DescriptorSetAllocateInfo info{
		.descriptorPool     = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &descriptor_set_layout,
	};

	CHECK_RESULT(device.allocateDescriptorSets(&info, &descriptor_set));
}

void SDFSample::CreateSwapchain() {
	int x, y, width, height;
	window.GetRect(x, y, width, height);
	VulkanRHI::SwapchainInfo info{
		.surface            = surface,
		.extent             = {.width = static_cast<u32>(width), .height = static_cast<u32>(height)},
		.queue_family_index = queue_family_index,
	};
	CHECK_RESULT(swapchain.Create(device, physical_device, info, GetAllocator()));
}

void SDFSample::CreatePipelineLayout() {
	vk::PushConstantRange push_constant_range{
		.stageFlags = vk::ShaderStageFlagBits::eFragment,
		.offset     = 0,
		.size       = physical_device.GetMaxPushConstantsSize(),
	};

	vk::PipelineLayoutCreateInfo info{
		.setLayoutCount         = 1,
		.pSetLayouts            = &descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &push_constant_range,
	};

	CHECK_RESULT(device.createPipelineLayout(&info, GetAllocator(), &pipeline_layout));
}

void SDFSample::CreatePipeline() {
	std::optional<std::vector<std::byte>> shader_codes[] = {
		ng::ReadBinaryFile("Shaders/Quad.vert.spv"),
		ng::ReadBinaryFile(bSlang ? "Shaders/SimpleSdf.slang.spv" : "Shaders/SimpleSdf.frag.spv"),
		// ng::ReadBinaryFile("Shaders/SimpleSdf.slang.spv"),
	};

	for (auto const& code : shader_codes) {
		if (!code.has_value()) {
			std::printf("Failed to read shader files!\n");
			std::exit(1);
		}
	}

	vk::ShaderModuleCreateInfo shader_module_infos[std::size(shader_codes)];
	vk::ShaderModule           shader_modules[std::size(shader_codes)];

	for (u32 i = 0; i < std::size(shader_codes); ++i) {
		shader_module_infos[i].codeSize = shader_codes[i]->size();
		shader_module_infos[i].pCode    = reinterpret_cast<const u32*>(shader_codes[i]->data());
		CHECK_RESULT(device.createShaderModule(&shader_module_infos[i], GetAllocator(), &shader_modules[i]));
	}

	vk::PipelineShaderStageCreateInfo shader_stages[] = {
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = shader_modules[0], .pName = "main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = shader_modules[1], .pName = "main"},
	};

	vk::VertexInputAttributeDescription vertex_input_attribute_descriptions[]{
		{
			.location = 0,
			.binding  = 0,
			.format   = vk::Format::eR32G32B32Sfloat,
			.offset   = 0,
		},
	};

	vk::VertexInputBindingDescription vertex_input_binding_description{
		.binding   = 0,
		.stride    = sizeof(Vertex),
		.inputRate = vk::VertexInputRate::eVertex,
	};

	vk::PipelineVertexInputStateCreateInfo vertex_input_state{};

	// vk::PipelineVertexInputStateCreateInfo vertex_input_state{
	// 	.vertexBindingDescriptionCount   = 1,
	// 	.pVertexBindingDescriptions      = &vertex_input_binding_description,
	// 	.vertexAttributeDescriptionCount = static_cast<u32>(std::size(vertex_input_attribute_descriptions)),
	// 	.pVertexAttributeDescriptions    = vertex_input_attribute_descriptions,
	// };

	vk::PipelineInputAssemblyStateCreateInfo input_assembly_state{
		.flags                  = {},
		.topology               = vk::PrimitiveTopology::eTriangleList,
		.primitiveRestartEnable = vk::False,
	};

	vk::PipelineViewportStateCreateInfo viewport_state{
		.viewportCount = 1,
		.scissorCount  = 1,
	};

	vk::PipelineRasterizationStateCreateInfo rasterization_state{
		// .cullMode  = vk::CullModeFlagBits::eBack,
		.frontFace = vk::FrontFace::eCounterClockwise,
		.lineWidth = 1.0f,
	};

	vk::PipelineMultisampleStateCreateInfo multisample_state{
		.rasterizationSamples = vk::SampleCountFlagBits::e1,
	};

	vk::PipelineDepthStencilStateCreateInfo depth_stencil_state{
		.depthTestEnable  = vk::True,
		.depthWriteEnable = vk::True,
		.depthCompareOp   = vk::CompareOp::eLess,
		.minDepthBounds   = 0.0f,
		.maxDepthBounds   = 1.0f,
	};

	vk::PipelineColorBlendAttachmentState color_blend_attachment_state{
		.blendEnable    = vk::False,
		.colorWriteMask = vk::ColorComponentFlagBits::eR |
						  vk::ColorComponentFlagBits::eG |
						  vk::ColorComponentFlagBits::eB |
						  vk::ColorComponentFlagBits::eA,
	};

	vk::PipelineColorBlendStateCreateInfo color_blend_state{
		.attachmentCount = 1,
		.pAttachments    = &color_blend_attachment_state,
	};

	vk::DynamicState dynamic_states[] = {
		vk::DynamicState::eViewport,
		vk::DynamicState::eScissor,
	};

	vk::PipelineDynamicStateCreateInfo dynamic_state{
		.dynamicStateCount = static_cast<u32>(std::size(dynamic_states)),
		.pDynamicStates    = dynamic_states,
	};

	vk::PipelineRenderingCreateInfo pipeline_rendering_info{
		.viewMask                = 0,
		.colorAttachmentCount    = 1,
		.pColorAttachmentFormats = &swapchain.GetFormat(),
		// .depthAttachmentFormat   = vk::Format::eD32Sfloat,
		// .stencilAttachmentFormat = vk::Format::eUndefined,
	};

	vk::GraphicsPipelineCreateInfo info{
		.pNext               = &pipeline_rendering_info,
		.stageCount          = static_cast<u32>(std::size(shader_stages)),
		.pStages             = shader_stages,
		.pVertexInputState   = &vertex_input_state,
		.pInputAssemblyState = &input_assembly_state,
		.pViewportState      = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState   = &multisample_state,
		// .pDepthStencilState  = &depth_stencil_state,
		.pColorBlendState = &color_blend_state,
		.pDynamicState    = &dynamic_state,
		.layout           = pipeline_layout,
	};

	CHECK_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &info, GetAllocator(), &pipeline));

	for (auto& shader_module : shader_modules) {
		device.destroyShaderModule(shader_module, GetAllocator());
	}
}

void SDFSample::CreateBuffers(std::size_t size) {

	// clang-format off
	CHECK_RESULT(staging_buffer.Create(device, vma_allocator, {
		.size   = size,
		.usage  = vk::BufferUsageFlagBits::eTransferSrc,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	}));

	CHECK_RESULT(sdf_weights_buffer.Create(device, vma_allocator, {
		.size   = size,
		.usage  = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		.memory = vk::MemoryPropertyFlagBits::eDeviceLocal,
	}));

	vk::DescriptorBufferInfo buffer_info{
		.buffer = sdf_weights_buffer,
		.offset = 0,
		.range  = size,
	};

	vk::WriteDescriptorSet write{
		.dstSet          = descriptor_set,
		.dstBinding      = 0,
		.dstArrayElement = 0,
		.descriptorCount = 1,
		.descriptorType  = vk::DescriptorType::eStorageBuffer,
		.pBufferInfo     = &buffer_info,
	};

	device.updateDescriptorSets(1, &write, 0, nullptr);
	// clang-format on
}

void SDFSample::UploadNetworkWeights() {
	auto ConvertOptimalLayer = [this](void const* src, std::size_t src_size, std::byte* dst, ng::Linear const& linear) {
		std::size_t expected_size = linear.GetWeightsSize();

		std::size_t required_size = expected_size;

		vk::ConvertCooperativeVectorMatrixInfoNV info{
			.srcSize          = src_size,
			.srcData          = {.hostAddress = src},
			.pDstSize         = &required_size,
			.dstData          = {.hostAddress = dst + linear.GetWeightsOffset()},
			.srcComponentType = kSrcComponentType,
			.dstComponentType = kDstMatrixType,
			.numRows          = linear.GetOutputsCount(),
			.numColumns       = linear.GetInputsCount(),
			.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
			.srcStride        = linear.GetInputsCount() * ng::GetVulkanComponentSize(kSrcComponentType),
			// .dstLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
			.dstLayout = kDstLayout,
			.dstStride = linear.GetInputsCount() * ng::GetVulkanComponentSize(kDstMatrixType),
		};

		info.dstData.hostAddress = nullptr;
		// CHECK_RESULT(device.convertCooperativeVectorMatrixNV(&info));
		vk::Result result;
		// result = device.convertCooperativeVectorMatrixNV(&info);
		CHECK_RESULT(device.convertCooperativeVectorMatrixNV(&info));
		if (result == vk::Result::eIncomplete) {
		}
		if (required_size != expected_size) {
			std::printf("Expected size: %zu, actual size: %zu\n", expected_size, required_size);
			std::exit(1);
		}
		info.dstData.hostAddress = dst + linear.GetWeightsOffset();
		CHECK_RESULT(device.convertCooperativeVectorMatrixNV(&info));
		// vk::Result result;
		// result = device.convertCooperativeVectorMatrixNV(&info);

		// do {
		// 	if (result < vk::Result::eSuccess) CHECK_RESULT(result);
		// } while (result == vk::Result::eIncomplete);

		// CHECK_RESULT(result);
	};

	auto* staging_ptr = reinterpret_cast<std::byte*>(staging_buffer.GetMappedData());

	ConvertOptimalLayer(kSDFWeights0, sizeof(kSDFWeights0), staging_ptr, network.GetLayer<ng::Linear>(0));
	ConvertOptimalLayer(kSDFWeights1, sizeof(kSDFWeights1), staging_ptr, network.GetLayer<ng::Linear>(1));
	ConvertOptimalLayer(kSDFWeights2, sizeof(kSDFWeights2), staging_ptr, network.GetLayer<ng::Linear>(2));
	ConvertOptimalLayer(kSDFWeights3, sizeof(kSDFWeights3), staging_ptr, network.GetLayer<ng::Linear>(3));

	std::memcpy(staging_ptr + network.GetLayer<ng::Linear>(0).GetBiasesOffset(), kSDFBias0, sizeof(kSDFBias0));
	std::memcpy(staging_ptr + network.GetLayer<ng::Linear>(1).GetBiasesOffset(), kSDFBias1, sizeof(kSDFBias1));
	std::memcpy(staging_ptr + network.GetLayer<ng::Linear>(2).GetBiasesOffset(), kSDFBias2, sizeof(kSDFBias2));
	std::memcpy(staging_ptr + network.GetLayer<ng::Linear>(3).GetBiasesOffset(), kSDFBias3, sizeof(kSDFBias3));

	vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	// clang-format off
	cmd.copyBuffer(staging_buffer, sdf_weights_buffer, {{
		.srcOffset = 0,
		.dstOffset = 0,
		.size      = network.GetParametersSize(),
	}});
	// clang-format on
	CHECK_RESULT(cmd.end());
	CHECK_RESULT(queue.submit({vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_RESULT(queue.waitIdle());
}

void SDFSample::DrawWindow() {
	auto HandleSwapchainResult = [this](vk::Result result) -> bool {
		switch (result) {
		case vk::Result::eSuccess:           return true;
		case vk::Result::eErrorOutOfDateKHR: swapchain_dirty = true; return false;
		case vk::Result::eSuboptimalKHR:     swapchain_dirty = true; return true;
		default:
			CHECK_RESULT(result);
		}
		return false;
	};
	CHECK_RESULT(device.waitForFences(1, &swapchain.GetCurrentFence(), vk::True, std::numeric_limits<u32>::max()));
	CHECK_RESULT(device.resetFences(1, &swapchain.GetCurrentFence()));
	device.resetCommandPool(swapchain.GetCurrentCommandPool());
	if (!HandleSwapchainResult(swapchain.AcquireNextImage())) return;
	RecordCommands();
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return;
}

void SDFSample::RecordCommands() {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	vk::Rect2D               render_rect{0, 0, static_cast<u32>(width), static_cast<u32>(height)};
	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	vk::Image swapchain_image = swapchain.GetCurrentImage();
	// cmd.SetViewport({0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f});
	cmd.SetViewport({0.0f, static_cast<float>(height), static_cast<float>(width), -static_cast<float>(height), 0.0f, 1.0f});
	cmd.SetScissor(render_rect);
	cmd.Barrier({
		.image         = swapchain_image,
		.aspectMask    = vk::ImageAspectFlagBits::eColor,
		.oldLayout     = vk::ImageLayout::eUndefined,
		.newLayout     = vk::ImageLayout::eColorAttachmentOptimal,
		.srcStageMask  = vk::PipelineStageFlagBits2::eNone,
		.srcAccessMask = vk::AccessFlagBits2::eNone,
		.dstStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
	});
	cmd.BeginRendering({
		.renderArea       = render_rect,
		.colorAttachments = {{{
			.imageView   = swapchain.GetCurrentImageView(),
			.imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
		}}},
	});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
	// std::printf("W offsets: %u %u %u %u\n", gWeightsOffsets[0], gWeightsOffsets[1], gWeightsOffsets[2], gWeightsOffsets[3]);
	// std::printf("B offsets: %u %u %u %u\n", gBiasOffsets[0], gBiasOffsets[1], gBiasOffsets[2], gBiasOffsets[3]);
	SDFConstants constants{
		.resolution      = {static_cast<float>(width), static_cast<float>(height)},
		.mouse           = {mouse.x, static_cast<float>(height) - mouse.y},
		.weights_offsets = {
			static_cast<u32>(network.GetLayer<ng::Linear>(0).GetWeightsOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(1).GetWeightsOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(2).GetWeightsOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(3).GetWeightsOffset()),
		},
		.bias_offsets = {
			static_cast<u32>(network.GetLayer<ng::Linear>(0).GetBiasesOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(1).GetBiasesOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(2).GetBiasesOffset()),
			static_cast<u32>(network.GetLayer<ng::Linear>(3).GetBiasesOffset()),
		},
	};
	cmd.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants), &constants);
	// vk::DeviceSize offsets[] = {0};
	// cmd.bindVertexBuffers(0, 1, &vertex_buffer, offsets);
	cmd.draw(std::size(kVertices), 1, 0, 0);
	cmd.endRendering();
	cmd.Barrier({
		.image         = swapchain_image,
		.aspectMask    = vk::ImageAspectFlagBits::eColor,
		.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout     = vk::ImageLayout::ePresentSrcKHR,
		.srcStageMask  = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
		.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
		.dstStageMask  = vk::PipelineStageFlagBits2::eNone,
		.dstAccessMask = vk::AccessFlagBits2::eNone,
	});
	CHECK_RESULT(cmd.end());
}

void SDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
	// std::printf("Recr with size %dx%d\n", width, height);
}

void SDFSample::Run() {

	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		DrawWindow();
	} while (true);
}

auto ParseArgs(int argc, char const* argv[]) -> char const* {
	for (std::string_view const arg : std::span(argv + 1, argc - 1)) {
		if (arg == "--slang") bSlang = true;
		else if (arg == "--verbose") bVerbose = true;
		else if (arg == "--validation") bValidation = true;
		else return arg.data();
	}
	return nullptr;
}

auto main(int argc, char const* argv[]) -> int {
	if (char const* unknown_arg = ParseArgs(argc, argv); unknown_arg) {
		std::printf("Unknown argument: %s\n", unknown_arg);
		std::printf("Usage: %ls [--slang] [--verbose] [--validation]\n", (std::filesystem::path(argv[0])).filename().c_str());
		return 0;
	}
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	SDFSample sample;
	gSDFSample = &sample;
	sample.Init();
	sample.Run();
	return 0;
}
