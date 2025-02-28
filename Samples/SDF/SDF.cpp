#include "../Common/Vulkan/CheckResult.h"
#include "Shaders/SDFConstants.h"

import NeuralGraphics;
import vulkan_hpp;
import std;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;

using namespace ng::Types;
using namespace ng::Utils;

#ifndef NDEBUG
bool constexpr kEnableValidationLayers = true;
#else
bool constexpr kEnableValidationLayers = false;
#endif

static constexpr u32 kApiVersion = vk::ApiVersion13;

static constexpr char const* kEnabledLayers[]           = {"VK_LAYER_KHRONOS_validation"};
static constexpr char const* kEnabledDeviceExtensions[] = {
	vk::KHRSwapchainExtensionName,
	// vk::NVCooperativeVectorExtensionName,
};

struct PhysicalDevice : public VulkanRHI::PhysicalDevice {
	bool IsSuitable(vk::SurfaceKHR const& surface) {
		bool const bSupportsExtensions = SupportsExtensions(kEnabledDeviceExtensions);
		bool const bSupportsQueues     = SupportsQueue({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
		// bool const bSupportsCooperativeVector = cooperative_vector_features.cooperativeVector == vk::True;

		if (bSupportsExtensions /* && bSupportsCooperativeVector  */ && bSupportsQueues) {
			return true;
		}
		return false;
	}

	vk::PhysicalDeviceCooperativeVectorFeaturesNV cooperative_vector_features{};

	u32 graphics_queue_family_index = std::numeric_limits<u32>::max();
};

struct Vertex {
	struct {
		float x, y, z;
	} position;
};

constexpr Vertex vertices[] = {
	Vertex{-1.0, -1.0, 0.0},
	Vertex{-1.0, 1.0, 0.0},
	Vertex{1.0, -1.0, 0.0},
	Vertex{-1.0, 1.0, 0.0},
	Vertex{1.0, 1.0, 0.0},
	Vertex{1.0, -1.0, 0.0},
};

class SDFSample {
public:
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

	void CreateVertexBuffer();

	void RecordCommands();
	void DrawWindow();
	void RecreateSwapchain(int width, int height);

	auto GetAllocator() -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() -> vk::PipelineCache { return pipeline_cache; }

	Window window{};
	struct {
		float x = 0.5f;
		float y = 0.5f;
	} mouse;

	vk::Instance                    instance{};
	vk::AllocationCallbacks const*  allocator{nullptr};
	VmaAllocator                    vma_allocator{};
	vk::PipelineCache               pipeline_cache{nullptr};
	vk::DebugUtilsMessengerEXT      debug_messenger{};
	std::span<char const* const>    enabled_layers = {};
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
};

SDFSample*  gSDFSample = nullptr;
static void FramebufferSizeCallback(Window* window, int width, int height) {
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

	// CreateVertexBuffer();
}

SDFSample::~SDFSample() { Destroy(); }

void SDFSample::Destroy() {

	if (device) {
		CHECK_RESULT(device.waitIdle());

		staging_buffer.Destroy();
		vertex_buffer.Destroy();

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
		if constexpr (kEnableValidationLayers) {
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
	if constexpr (kEnableValidationLayers) {
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
		.pNext                   = kEnableValidationLayers ? &kDebugUtilsCreateInfo : nullptr,
		.pApplicationInfo        = &applicationInfo,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(enabledExtensions)),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};
	CHECK_RESULT(vk::createInstance(&info, GetAllocator(), &instance));

	if (kEnableValidationLayers) {
		LoadInstanceDebugUtilsFunctionsEXT(instance);
		CHECK_RESULT(instance.createDebugUtilsMessengerEXT(&kDebugUtilsCreateInfo, allocator, &debug_messenger));
	}

	vk::Result result;
	std::tie(result, vulkan_physical_devices) = instance.enumeratePhysicalDevices();
	CHECK_RESULT(result);
}

void SDFSample::SelectPhysicalDevice() {
	AddToPNext(physical_device.GetFeatures2(), physical_device.cooperative_vector_features);
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface)) {
			return;
		}
	}

	std::printf("No suitable physical device found\n");
	// std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	std::exit(1);
}

void SDFSample::GetPhysicalDeviceInfo() {
}

void SDFSample::CreateDevice() {
	float const queue_priorities[] = {1.0f};

	auto [result, index] = physical_device.GetQueueFamilyIndex({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
	if (result != vk::Result::eSuccess || !index.has_value()) {
		std::printf("Failed to get graphics queue family index\n");
		std::exit(1);
	}

	queue_family_index = index.value();

	vk::DeviceQueueCreateInfo queue_create_infos[] = {
		{
			.queueFamilyIndex = queue_family_index,
			.queueCount       = 1,
			.pQueuePriorities = queue_priorities,
		}};

	std::span enabled_extensions = kEnabledDeviceExtensions;

	vk::PhysicalDeviceFeatures2 features2;

	vk::PhysicalDeviceVulkan11Features vulkan11{
		.storageBuffer16BitAccess = vk::True,
	};

	vk::PhysicalDeviceVulkan12Features vulkan12{
		.shaderFloat16       = vk::True,
		.bufferDeviceAddress = vk::True,
	};

	vk::PhysicalDeviceVulkan13Features vulkan13{
		.synchronization2 = vk::True,
		.dynamicRendering = vk::True,
	};

	vk::PhysicalDeviceCooperativeVectorFeaturesNV cooperative_vector_features{
		.cooperativeVector = vk::True,
	};

	features2.pNext = &vulkan11;
	vulkan11.pNext  = &vulkan12;
	vulkan12.pNext  = &vulkan13;
	vulkan13.pNext  = &cooperative_vector_features;

	vk::DeviceCreateInfo info{
		.pNext                   = &features2,
		.queueCreateInfoCount    = static_cast<u32>(std::size(queue_create_infos)),
		.pQueueCreateInfos       = queue_create_infos,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(enabled_extensions)),
		.ppEnabledExtensionNames = enabled_extensions.data(),
	};
	CHECK_RESULT(physical_device.createDevice(&info, GetAllocator(), &device));

	for (auto& info : queue_create_infos) {
		for (u32 index = 0; index < info.queueCount; ++index) {
			queue = device.getQueue(info.queueFamilyIndex, index);
		}
	}
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
		.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
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
		{vk::DescriptorType::eCombinedImageSampler, 1},
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
		.extent             = {u32(width), u32(height)},
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
		// ng::ReadBinaryFile("Shaders/SimpleSdf.vert.spv"),
		// ng::ReadBinaryFile("Shaders/SimpleSdfDebug.frag.spv"),
		ng::ReadBinaryFile("Shaders/Quad.vert.spv"),
		// ng::ReadBinaryFile("Shaders/SimpleSdfDebug.frag.spv"),
		ng::ReadBinaryFile("Shaders/SimpleSdf.frag.spv"),
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

void SDFSample::CreateVertexBuffer() {
	// clang-format off
	CHECK_RESULT(staging_buffer.Create(device, vma_allocator, {
		.size   = sizeof(vertices),
		.usage  = vk::BufferUsageFlagBits::eTransferSrc,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	}));

	CHECK_RESULT(vertex_buffer.Create(device, vma_allocator, {
		.size   = sizeof(vertices),
		.usage  = vk::BufferUsageFlagBits::eVertexBuffer |
				  vk::BufferUsageFlagBits::eTransferDst |
				  vk::BufferUsageFlagBits::eShaderDeviceAddress,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
	}));

	std::memcpy(staging_buffer.GetMappedData(), vertices, sizeof(vertices));

	vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	cmd.copyBuffer(staging_buffer, vertex_buffer, {{
		.srcOffset = 0,
		.dstOffset = 0,
		.size      = sizeof(vertices),
	}});
	CHECK_RESULT(cmd.end());
	CHECK_RESULT(queue.submit({vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_RESULT(queue.waitIdle());
	// clang-format on
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
		.image      = swapchain_image,
		.aspectMask = vk::ImageAspectFlagBits::eColor,
		.oldLayout  = vk::ImageLayout::eUndefined,
		.newLayout  = vk::ImageLayout::eColorAttachmentOptimal,
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
			.loadOp      = vk::AttachmentLoadOp::eClear,
			.storeOp     = vk::AttachmentStoreOp::eStore,
			.clearValue  = {.color = {.float32 = {{0.2f, 0.2f, 0.2f, 1.0f}}}},
		}}},
	});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	// cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
	SDFConstants constants{
		.resolution = {static_cast<float>(width), static_cast<float>(height)},
		.mouse      = {mouse.x, static_cast<float>(height) - mouse.y},
	};
	cmd.pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants), &constants);
	// vk::DeviceSize offsets[] = {0};
	// cmd.bindVertexBuffers(0, 1, &vertex_buffer, offsets);
	cmd.draw(std::size(vertices), 1, 0, 0);
	cmd.endRendering();
	cmd.Barrier({
		.image      = swapchain_image,
		.aspectMask = vk::ImageAspectFlagBits::eColor,
		.oldLayout  = vk::ImageLayout::eColorAttachmentOptimal,
		.newLayout  = vk::ImageLayout::ePresentSrcKHR,
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

		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;

		if (swapchain_dirty) {
			std::printf("Error: Swapchain is dirty\n");
			std::exit(1);
		}
		DrawWindow();
		// break;
	} while (!window.GetShouldClose());
}

int main(int argc, char const* argv[]) {
	std::filesystem::current_path(std::filesystem::path(argv[0]).parent_path());
	SDFSample sample;
	gSDFSample = &sample;
	sample.Init();
	sample.Run();
	return 0;
}
