#include "CheckResult.h"
#include "Shaders/SDFConfig.h"
#include "Shaders/SDFConstants.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;
import std;

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t
#include "Shaders/SDFWeights.h"

using namespace Utils;



class SDFSample {
public:
	static constexpr u32 kApiVersion = vk::ApiVersion13;

	static constexpr char const* kEnabledLayers[] = {
		"VK_LAYER_KHRONOS_validation",
	};
	static constexpr char const* kEnabledDeviceExtensions[] = {
		vk::KHRSwapchainExtensionName,
		vk::NVCooperativeVectorExtensionName,
		vk::NVCooperativeVectorExtensionName,
		vk::EXTShaderReplicatedCompositesExtensionName,
	};

	vk::ComponentTypeKHR kSrcComponentType = vk::ComponentTypeKHR::eFloat16;
	vk::ComponentTypeKHR kDstMatrixType    = vk::ComponentTypeKHR::eFloat16;
	vk::ComponentTypeKHR kDstVectorType    = vk::ComponentTypeKHR::eFloat16;

	static constexpr u32 kFramesInFlight = 3;

	static constexpr u32 kNetworkLayers = 4;

	struct NetworkOffsets {
		u32 weights_offsets[kNetworkLayers];
		u32 biases_offsets[kNetworkLayers];
	};

	~SDFSample();

	void Init();
	void Run();
	void RunTest();
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

	void CreatePipelines();
	[[nodiscard]]
	auto CreatePipeline(
		vk::ShaderModule vertex_shader_module,
		vk::ShaderModule fragment_shader_module,
		SdfFunctionType  function_type = SdfFunctionType::eCoopVec) -> vk::Pipeline;

	// void BuildNetwork();
	void CreateAndUploadBuffers();
	void WriteNetworkWeights(VulkanCoopVecNetwork const&         network,
							 VulkanRHI::Buffer const&            staging_buffer,
							 std::size_t                         staging_offset,
							 vk::CooperativeVectorMatrixLayoutNV dst_layout);

	void RecordCommands(vk::Pipeline pipeline, NetworkOffsets const& offsets);
	// Return time in nanoseconds
	auto GetQueryResult() -> u64;
	auto DrawWindow(vk::Pipeline pipeline, NetworkOffsets const& offsets) -> u64;
	void RecreateSwapchain(int width, int height);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }

	void SetTestMode(bool value) { bIsTestMode = value; }
	bool IsTestMode() const { return bIsTestMode; }
	void SetVerbose(bool value) { bVerbose = value; }
	void SetUseValidation(bool value) { bUseValidation = value; }

	bool bIsTestMode    = false;
	bool bVerbose       = false;
	bool bUseValidation = false;

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

	bool timestamps_supported = false;

	vk::DescriptorSetLayout descriptor_set_layout{};
	vk::DescriptorPool      descriptor_pool{};
	vk::DescriptorSet       descriptor_set{};

	vk::PipelineLayout pipeline_layout{};

	// vk::Pipeline coopvec_pipeline;
	// vk::Pipeline scalar_inline_pipeline;
	// vk::Pipeline scalar_buffer_pipeline;
	// vk::Pipeline vec4_pipeline;

	std::array<vk::Pipeline, u32(SdfFunctionType::eCount)> pipelines;

	VulkanRHI::Buffer staging_buffer{};
	VulkanRHI::Buffer sdf_weights_buffer{};

	VulkanCoopVecNetwork sdf_network = {
		Linear(3, 16),
		Linear(16, 16),
		Linear(16, 16),
		Linear(16, 1),
	};

	NetworkOffsets optimal_offsets;
	NetworkOffsets row_major_offsets;

	// std::vector<std::byte> network_parameters;

	static constexpr u32 kTimestampsPerFrame = 2;

	std::array<u64, kFramesInFlight * kTimestampsPerFrame>  timestamp_results = {};
	std::array<bool, kFramesInFlight * kTimestampsPerFrame> timestamp_valid   = {};

	auto GetCurrentTimestampResult() -> u64* {
		return timestamp_results.data() + swapchain.GetCurrentFrameIndex() * kTimestampsPerFrame;
	}

	auto GetTimestampResult(u32 frame_index) -> u64* {
		return timestamp_results.data() + frame_index * kTimestampsPerFrame;
	}

	auto GetCurrentTimestampIndex() -> u32 { return swapchain.GetCurrentFrameIndex() * kTimestampsPerFrame; }

	vk::QueryPool timestamp_query_pool{};
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
	gSDFSample->DrawWindow(gSDFSample->pipelines[u32(SdfFunctionType::eCoopVec)], gSDFSample->optimal_offsets);
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
	if (!bIsTestMode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}
	window.GetInputCallbacks().cursorPosCallback = CursorPosCallback;
	window.GetInputCallbacks().keyCallback       = KeyCallback;
	CreateInstance();

	CHECK_VULKAN_RESULT(WindowManager::CreateWindowSurface(
		instance, reinterpret_cast<GLFWwindow*>(window.GetHandle()),
		GetAllocator(), &surface));

	SelectPhysicalDevice();
	GetPhysicalDeviceInfo();

	CreateDevice();
	CreateVmaAllocator();

	CreateDescriptorSetLayout();
	CreateDescriptorPool();
	CreateDescriptorSet();
	CreateSwapchain();
	CreatePipelineLayout();

	LoadInstanceCooperativeVectorFunctionsNV(instance);
	CreateAndUploadBuffers();

	CreatePipelines();

	// Create timestamp query pool
	vk::QueryPoolCreateInfo query_pool_info{
		.flags      = {},
		.queryType  = vk::QueryType::eTimestamp,
		.queryCount = static_cast<u32>(std::size(timestamp_results)),
	};
	CHECK_VULKAN_RESULT(device.createQueryPool(&query_pool_info, GetAllocator(), &timestamp_query_pool));

	// network_parameters.resize(network.GetParametersSize());

	auto [result, cooperative_vector_properties] = physical_device.getCooperativeVectorPropertiesNV();
	CHECK_VULKAN_RESULT(result);

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
		CHECK_VULKAN_RESULT(device.waitIdle());

		if (timestamp_query_pool) {
			device.destroyQueryPool(timestamp_query_pool, GetAllocator());
			timestamp_query_pool = nullptr;
		}

		staging_buffer.Destroy();
		sdf_weights_buffer.Destroy();

		for (vk::Pipeline& pipeline : pipelines) {
			device.destroyPipeline(pipeline, GetAllocator());
			pipeline = vk::Pipeline{};
		}
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
		if (debug_messenger) {
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
	if (bUseValidation) {
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
					   //    vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
					   vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		.pfnUserCallback = DebugUtilsCallback,
		.pUserData       = nullptr,
	};

	vk::InstanceCreateInfo info{
		.pNext                   = bUseValidation ? &kDebugUtilsCreateInfo : nullptr,
		.pApplicationInfo        = &applicationInfo,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(enabledExtensions)),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};
	CHECK_VULKAN_RESULT(vk::createInstance(&info, GetAllocator(), &instance));

	if (bUseValidation) {
		LoadInstanceDebugUtilsFunctionsEXT(instance);
		CHECK_VULKAN_RESULT(instance.createDebugUtilsMessengerEXT(&kDebugUtilsCreateInfo, allocator, &debug_messenger));
	}

	vk::Result result;
	std::tie(result, vulkan_physical_devices) = instance.enumeratePhysicalDevices();
	CHECK_VULKAN_RESULT(result);
}

void SDFSample::SelectPhysicalDevice() {
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_VULKAN_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface, kEnabledDeviceExtensions)) {
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
		CHECK_VULKAN_RESULT(result);
	}

	queue_family_index = index.value();

	auto queue_family_properties = physical_device.GetQueueFamilyProperties(queue_family_index);
	timestamps_supported         = queue_family_properties.timestampValidBits > 0;

	vk::DeviceQueueCreateInfo queue_create_infos[] = {
		{
			.queueFamilyIndex = queue_family_index,
			.queueCount       = 1,
			.pQueuePriorities = queue_priorities,
		}};

	vk::StructureChain features{
		vk::PhysicalDeviceFeatures2{},
		vk::PhysicalDeviceVulkan11Features{.storageBuffer16BitAccess = vk::True},
		vk::PhysicalDeviceVulkan12Features{
			.shaderFloat16       = vk::True,
			.hostQueryReset      = timestamps_supported ? vk::True : vk::False,
			.bufferDeviceAddress = vk::True,
		},
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

	CHECK_VULKAN_RESULT(physical_device.createDevice(&info, GetAllocator(), &device));
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
	CHECK_VULKAN_RESULT(vk::Result(vmaCreateAllocator(&info, &vma_allocator)));
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

	CHECK_VULKAN_RESULT(device.createDescriptorSetLayout(&info, GetAllocator(), &descriptor_set_layout));
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

	CHECK_VULKAN_RESULT(device.createDescriptorPool(&info, GetAllocator(), &descriptor_pool));
}

void SDFSample::CreateDescriptorSet() {
	vk::DescriptorSetAllocateInfo info{
		.descriptorPool     = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &descriptor_set_layout,
	};

	CHECK_VULKAN_RESULT(device.allocateDescriptorSets(&info, &descriptor_set));
}

void SDFSample::CreateSwapchain() {
	int x, y, width, height;
	window.GetRect(x, y, width, height);
	VulkanRHI::SwapchainInfo info{
		.surface            = surface,
		.extent             = {.width = static_cast<u32>(width), .height = static_cast<u32>(height)},
		.queue_family_index = queue_family_index,
		.frames_in_flight   = kFramesInFlight,
	};
	CHECK_VULKAN_RESULT(swapchain.Create(device, physical_device, info, GetAllocator()));
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

	CHECK_VULKAN_RESULT(device.createPipelineLayout(&info, GetAllocator(), &pipeline_layout));
}

void SDFSample::CreatePipelines() {
	std::optional<std::vector<std::byte>> shader_codes[] = {
		ReadBinaryFile("Shaders/Quad.vert.spv"),
		ReadBinaryFile("Shaders/SdfMain.slang.spv"),
	};

	for (auto const code : shader_codes) {
		if (!code.has_value()) {
			std::printf("Failed to read shader file!\n");
			std::exit(1);
		}
	}

	vk::ShaderModuleCreateInfo shader_module_infos[std::size(shader_codes)];
	vk::ShaderModule           shader_modules[std::size(shader_codes)];

	for (u32 i = 0; i < std::size(shader_codes); ++i) {
		shader_module_infos[i].codeSize = shader_codes[i].value().size();
		shader_module_infos[i].pCode    = reinterpret_cast<const u32*>(shader_codes[i].value().data());
		CHECK_VULKAN_RESULT(device.createShaderModule(&shader_module_infos[i], GetAllocator(), &shader_modules[i]));
	}

	pipelines[int(SdfFunctionType::eCoopVec)]      = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eCoopVec);
	pipelines[int(SdfFunctionType::eScalarInline)] = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eScalarInline);
	pipelines[int(SdfFunctionType::eScalarBuffer)] = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eScalarBuffer);
	pipelines[int(SdfFunctionType::eVec4)]         = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eVec4);

	for (auto& shader_module : shader_modules) {
		device.destroyShaderModule(shader_module, GetAllocator());
	}
}

auto SDFSample::CreatePipeline(vk::ShaderModule vertex_shader_module,
							   vk::ShaderModule fragment_shader_module,
							   SdfFunctionType  function_type) -> vk::Pipeline {

	// Specialization constant for type of inferencing function
	SdfFunctionType specialization_value = function_type;

	vk::SpecializationMapEntry specialization_entry{
		.constantID = 0,
		.offset     = 0,
		.size       = sizeof(SdfFunctionType),
	};
	vk::SpecializationInfo specialization_info{
		.mapEntryCount = 1,
		.pMapEntries   = &specialization_entry,
		.dataSize      = sizeof(SdfFunctionType),
		.pData         = &specialization_value,
	};

	vk::PipelineShaderStageCreateInfo shader_stages[] = {
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = vertex_shader_module, .pName = "main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = fragment_shader_module, .pName = "main", .pSpecializationInfo = &specialization_info},
	};

	vk::VertexInputAttributeDescription vertex_input_attribute_descriptions[] = {{
		.location = 0,
		.binding  = 0,
		.format   = vk::Format::eR32G32B32Sfloat,
		.offset   = 0,
	}};

	vk::PipelineVertexInputStateCreateInfo vertex_input_state{};

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

	vk::Pipeline pipeline;
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &info, GetAllocator(), &pipeline));

	return pipeline;
}



void SDFSample::CreateAndUploadBuffers() {
	// BuildNetwork();
	std::size_t optimal_size_bytes   = 0;
	std::size_t row_major_size_bytes = 0;

	// Optimal layout
	CHECK_VULKAN_RESULT(sdf_network.UpdateOffsetsAndSize(
		device, vk::CooperativeVectorMatrixLayoutNV::eInferencingOptimal,
		kDstMatrixType, kDstVectorType));
	optimal_size_bytes = AlignTo(sdf_network.GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	if (bVerbose) sdf_network.Print();

	// Write optimal offsets
	for (u32 i = 0; i < kNetworkLayers; ++i) {
		u32 weights_offset = static_cast<u32>(sdf_network.GetLayer<Linear>(i).GetWeightsOffset());
		u32 biases_offset  = static_cast<u32>(sdf_network.GetLayer<Linear>(i).GetBiasesOffset());

		optimal_offsets.weights_offsets[i] = weights_offset;
		optimal_offsets.biases_offsets[i]  = biases_offset;
	}

	// Row-major layout
	CHECK_VULKAN_RESULT(sdf_network.UpdateOffsetsAndSize(
		device, vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
		kDstMatrixType, kDstVectorType));
	row_major_size_bytes = AlignTo(sdf_network.GetParametersSize(), CoopVecUtils::GetMatrixAlignment());

	if (bVerbose) sdf_network.Print();

	// Write row-major offsets
	for (u32 i = 0; i < kNetworkLayers; ++i) {
		u32 weights_offset = static_cast<u32>(sdf_network.GetLayer<Linear>(i).GetWeightsOffset());
		u32 biases_offset  = static_cast<u32>(sdf_network.GetLayer<Linear>(i).GetBiasesOffset());

		row_major_offsets.weights_offsets[i] = optimal_size_bytes + weights_offset;
		row_major_offsets.biases_offsets[i]  = optimal_size_bytes + biases_offset;
	}

	std::size_t total_size_bytes = optimal_size_bytes + row_major_size_bytes;

	// clang-format off
	CHECK_VULKAN_RESULT(sdf_weights_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
		.memory = vk::MemoryPropertyFlagBits::eDeviceLocal,
	}));

	CHECK_VULKAN_RESULT(staging_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferSrc,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	}));
	// clang-format on

	// Update descriptor
	{
		vk::DescriptorBufferInfo buffer_info{
			.buffer = sdf_weights_buffer,
			.offset = 0,
			.range  = total_size_bytes,
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
	}

	// Upload weights
	{
		auto update_and_write = [&](u32 offset, vk::CooperativeVectorMatrixLayoutNV layout) {
			CHECK_VULKAN_RESULT(sdf_network.UpdateOffsetsAndSize(device, layout, kDstMatrixType, kDstVectorType))
			WriteNetworkWeights(sdf_network, staging_buffer, offset, layout);
		};
		std::size_t offset = 0;
		update_and_write(offset, vk::CooperativeVectorMatrixLayoutNV::eInferencingOptimal);
		offset += optimal_size_bytes;
		update_and_write(offset, vk::CooperativeVectorMatrixLayoutNV::eRowMajor);

		vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
		CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
		// clang-format off
		cmd.copyBuffer(staging_buffer, sdf_weights_buffer, {{
			.srcOffset = 0,
			.dstOffset = 0,
			.size      = total_size_bytes,
		}});
		// clang-format on
		CHECK_VULKAN_RESULT(cmd.end());
		CHECK_VULKAN_RESULT(queue.submit({vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
		CHECK_VULKAN_RESULT(queue.waitIdle());
	}
}

void SDFSample::WriteNetworkWeights(VulkanCoopVecNetwork const&         network,
									VulkanRHI::Buffer const&            staging_buffer,
									std::size_t                         staging_offset,
									vk::CooperativeVectorMatrixLayoutNV dst_layout) {
	auto ConvertLayerWeights = [this, dst_layout](void const* src, std::size_t src_size,
												  std::byte* dst, Linear const& linear) {
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
			.srcStride        = linear.GetInputsCount() * GetVulkanComponentSize(kSrcComponentType),
			.dstLayout        = dst_layout,
			.dstStride        = linear.GetInputsCount() * GetVulkanComponentSize(kDstMatrixType),
		};

		info.dstData.hostAddress = nullptr;
		CHECK_VULKAN_RESULT(device.convertCooperativeVectorMatrixNV(&info));
		if (required_size != expected_size) {
			std::printf("Expected size: %zu, actual size: %zu\n", expected_size, required_size);
			std::exit(1);
		}
		info.dstData.hostAddress = dst + linear.GetWeightsOffset();
		CHECK_VULKAN_RESULT(device.convertCooperativeVectorMatrixNV(&info));
	};

	std::byte* staging_ptr = staging_offset + reinterpret_cast<std::byte*>(staging_buffer.GetMappedData());

	ConvertLayerWeights(kSDFWeights0, sizeof(kSDFWeights0), staging_ptr, network.GetLayer<Linear>(0));
	ConvertLayerWeights(kSDFWeights1, sizeof(kSDFWeights1), staging_ptr, network.GetLayer<Linear>(1));
	ConvertLayerWeights(kSDFWeights2, sizeof(kSDFWeights2), staging_ptr, network.GetLayer<Linear>(2));
	ConvertLayerWeights(kSDFWeights3, sizeof(kSDFWeights3), staging_ptr, network.GetLayer<Linear>(3));

	std::memcpy(staging_ptr + network.GetLayer<Linear>(0).GetBiasesOffset(), kSDFBias0, sizeof(kSDFBias0));
	std::memcpy(staging_ptr + network.GetLayer<Linear>(1).GetBiasesOffset(), kSDFBias1, sizeof(kSDFBias1));
	std::memcpy(staging_ptr + network.GetLayer<Linear>(2).GetBiasesOffset(), kSDFBias2, sizeof(kSDFBias2));
	std::memcpy(staging_ptr + network.GetLayer<Linear>(3).GetBiasesOffset(), kSDFBias3, sizeof(kSDFBias3));
}

auto SDFSample::GetQueryResult() -> u64 {
	vk::Result result =
		device.getQueryPoolResults(
			timestamp_query_pool,
			GetCurrentTimestampIndex(),
			kTimestampsPerFrame,
			sizeof(timestamp_results[0]) * kTimestampsPerFrame,
			GetCurrentTimestampResult(),
			sizeof(u64),
			vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);

	switch (result) {
	case vk::Result::eSuccess: {
		u64* current_timestamps = GetCurrentTimestampResult();
		u64  start              = current_timestamps[0];
		u64  end                = current_timestamps[1];
		u64  elapsed            = end > start ? end - start : 0;
		return elapsed;
		break;
	}
	case vk::Result::eNotReady:
		// timestamp_valid = false;
		std::printf("Timestamp not ready\n");
		break;
	default:
		CHECK_VULKAN_RESULT(result);
	}
	return 0ull;
}

auto SDFSample::DrawWindow(vk::Pipeline pipeline, NetworkOffsets const& offsets) -> u64 {
	auto HandleSwapchainResult = [this](vk::Result result) -> bool {
		switch (result) {
		case vk::Result::eSuccess:           return true;
		case vk::Result::eErrorOutOfDateKHR: swapchain_dirty = true; return false;
		case vk::Result::eSuboptimalKHR:     swapchain_dirty = true; return true;
		default:
			CHECK_VULKAN_RESULT(result);
		}
		return false;
	};
	CHECK_VULKAN_RESULT(device.waitForFences(1, &swapchain.GetCurrentFence(), vk::True, std::numeric_limits<u32>::max()));
	CHECK_VULKAN_RESULT(device.resetFences(1, &swapchain.GetCurrentFence()));
	device.resetCommandPool(swapchain.GetCurrentCommandPool());
	if (!HandleSwapchainResult(swapchain.AcquireNextImage())) return 0ull;
	RecordCommands(pipeline, offsets);
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;

	// Call it here
	u64 elapsed = GetQueryResult();
	swapchain.EndFrame();
	return elapsed;
}

void SDFSample::RecordCommands(vk::Pipeline pipeline, NetworkOffsets const& offsets) {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	vk::Rect2D               render_rect{0, 0, static_cast<u32>(width), static_cast<u32>(height)};
	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	cmd.resetQueryPool(timestamp_query_pool, GetCurrentTimestampIndex(), kTimestampsPerFrame);
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timestamp_query_pool, GetCurrentTimestampIndex());

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
	// std::printf("W offsets: %u %u %u %u\n", offsets.weights_offsets[0], offsets.weights_offsets[1], offsets.weights_offsets[2], offsets.weights_offsets[3]);
	// std::printf("B offsets: %u %u %u %u\n", offsets.biases_offsets[0], offsets.biases_offsets[1], offsets.biases_offsets[2], offsets.biases_offsets[3]);
	SDFConstants constants{
		.resolution = {static_cast<float>(width), static_cast<float>(height)},
		.mouse      = {mouse.x, static_cast<float>(height) - mouse.y},
	};
	for (int i = 0; i < kNetworkLayers; ++i) {
		constants.weights_offsets[i] = offsets.weights_offsets[i];
		constants.bias_offsets[i]    = offsets.biases_offsets[i];
	}

	cmd.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants), &constants);
	u32 vertex_count = 6u;
	cmd.draw(vertex_count, 1, 0, 0);
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

	// Write timestamp at end
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, timestamp_query_pool, GetCurrentTimestampIndex() + 1);
	CHECK_VULKAN_RESULT(cmd.end());
}

void SDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}
static auto last_frame_time = std::chrono::steady_clock::now();

void SDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		DrawWindow(pipelines[u32(SdfFunctionType::eCoopVec)], optimal_offsets);
	} while (true);
}

void SDFSample::RunTest() {
	struct TestData {
		vk::Pipeline   pipeline;
		NetworkOffsets offsets;
	};
	// WindowManager::WaitEvents();
	// if (window.GetShouldClose()) return;

	window.SetWindowMode(WindowMode::eFullscreen);
	int x, y, width, height;
	window.GetRect(x, y, width, height);
	// std::printf("Resizing to %dx%d\n", width, height);
	RecreateSwapchain(width, height);
	// DrawWindow();
	constexpr u32 kNumTestRuns  = 1000;
	constexpr u32 kNumTestKinds = u32(SdfFunctionType::eCount);

	TestData test_data[kNumTestKinds] = {
		{pipelines[u32(SdfFunctionType::eCoopVec)], optimal_offsets},
		{pipelines[u32(SdfFunctionType::eScalarInline)], row_major_offsets},
		{pipelines[u32(SdfFunctionType::eScalarBuffer)], row_major_offsets},
		{pipelines[u32(SdfFunctionType::eVec4)], row_major_offsets},
	};

	std::vector<std::array<u64, kNumTestKinds>> test_times(kNumTestRuns);

	for (u32 test_kind_index = 0; test_kind_index < kNumTestKinds; ++test_kind_index) {
		TestData& data = test_data[test_kind_index];
		for (u32 iter = 0; iter < kNumTestRuns; ++iter) {
			// WindowManager::PollEvents();
			u64   time_nanoseconds = DrawWindow(data.pipeline, data.offsets);
			float ns_per_tick      = physical_device.GetNsPerTick();
			float elapsed_ms       = (time_nanoseconds * ns_per_tick) / 1e6f;
			test_times[iter][test_kind_index] = time_nanoseconds;
		}
	}

	// Print csv
	std::printf("CoopVec, ScalarInline, ScalarBuffer, Vec4 \n");
	for (u32 iter = 0; iter < kNumTestRuns; ++iter) {
		for (u32 test_kind_index = 0; test_kind_index < kNumTestKinds - 1; ++test_kind_index) {
			std::printf("%llu, ", test_times[iter][test_kind_index]);
		}
		std::printf("%llu \n", test_times[iter][kNumTestKinds - 1]);
	}
}

auto ParseArgs(int argc, char const* argv[]) -> char const* {
	for (std::string_view const arg : std::span(argv + 1, argc - 1)) {
		if (arg == "--test" || arg == "-t") gSDFSample->SetTestMode(true);
		else if (arg == "--verbose") gSDFSample->SetVerbose(true);
		else if (arg == "--validation") gSDFSample->SetUseValidation(true);
		else return arg.data();
	}
	return nullptr;
}

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	SDFSample sample;
	gSDFSample = &sample;

	if (char const* unknown_arg = ParseArgs(argc, argv); unknown_arg) {
		std::printf("Unknown argument: %s\n", unknown_arg);
		std::printf("Usage: %ls [--test] [--verbose] [--validation]\n",
					std::filesystem::path(argv[0]).filename().c_str());
		return 0;
	}

	sample.Init();
	if (sample.IsTestMode()) {
		sample.RunTest();
	} else {
		sample.Run();
	}
	return 0;
}
