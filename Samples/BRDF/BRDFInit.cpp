module;

#include "stddef.h" // offsetof

module BRDFSample;

#include "CheckResult.h"
#include "Shaders/BRDFConfig.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import WeightsLoader;
import SamplesCommon;
import Math;
import std;

namespace {
static void FramebufferSizeCallback(GLFWWindow* window, int width, int height) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	sample->swapchain_dirty = true;
	if (width <= 0 || height <= 0) return;
	sample->RecreateSwapchain(width, height);

	sample->camera.updateProjection(width, height);
}

static void WindowRefreshCallback(GLFWWindow* window) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	int x, y, width, height;
	window->GetRect(x, y, width, height);
	if (width <= 0 || height <= 0) return;
	sample->DrawWindow();
}

static void CursorPosCallback(GLFWWindow* window, double xpos, double ypos) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	sample->mouse.delta_x = static_cast<float>(xpos - sample->mouse.x);
	sample->mouse.delta_y = -static_cast<float>(ypos - sample->mouse.y);
	sample->mouse.x       = static_cast<float>(xpos);
	sample->mouse.y       = static_cast<float>(ypos);

	ProcessViewportInput(sample->window, sample->camera, sample->mouse, sample->mouse.delta_x, sample->mouse.delta_y);

	// std::printf("mouse position: (%f, %f), delta: (%f, %f)\n", sample->mouse.x, sample->mouse.y,
	// 			sample->mouse.delta_x, sample->mouse.delta_y);
}

static void KeyCallback(GLFWWindow* window, int key, int scancode, int action, int mods) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	using namespace Glfw;
	if (Action(action) == Action::ePress) {
		switch (Key(key)) {
		case Key::eEscape:
			window->SetShouldClose(true);
			break;
		case Key::eF8: {
			// sample->pending_image_save = true;

			// auto fname = "sdf.bmp";

			char fname[256] = {};

			std::snprintf(fname, sizeof(fname), "brdf_%d.bmp", *sample->function_id);

			sample->SaveSwapchainImageToFile(fname);
			// sample->SaveSwapchainImageToFile("sdf.png");
		} break;
		default: break;
		}
	}
}

static void MouseButtonCallback(GLFWWindow* in_window, int in_button, int in_action, int in_mods) {
	BRDFSample* sample = static_cast<BRDFSample*>(in_window->GetUserPointer());
	using namespace Glfw;

	MouseButton button = static_cast<MouseButton>(in_button);
	Action      action = static_cast<Action>(in_action);
	Mod         mods   = static_cast<Mod>(in_mods);

	sample->mouse.button_state[in_button] = action;

	if (button == MouseButton::eRight) {
		if (action == Action::ePress) {
			if (mods == Mod::eAlt) {
				sample->mouse.StartDragging();
			}
		} else if (action == Action::eRelease) {
			sample->mouse.StopDragging();
		}
	}
}
} // namespace
void BRDFSample::Init() {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();
	u32 const initial_width = 1600, initial_height = 1200;

	window.Init({.x = 30, .y = 30, .width = initial_width, .height = initial_height, .title = "BRDF Sample"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!is_test_mode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}

	int x, y, width, height;
	window.GetFullScreenRect(x, y, width, height);

	camera.fov = 35.0f;
	camera.updateProjection(initial_width, initial_height);

	window.GetInputCallbacks().cursorPosCallback   = CursorPosCallback;
	window.GetInputCallbacks().keyCallback         = KeyCallback;
	window.GetInputCallbacks().mouseButtonCallback = MouseButtonCallback;
	window.SetUserPointer(this);

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
	CreateAndUploadBuffers({.file_name = weights_file_name.size() > 0 ? weights_file_name : "Assets/simple_brdf_weights.bin", .header = ""});

	depth_image.Create(
		device, vma_allocator, allocator,
		{.image_info = {
			 .flags         = {},
			 .imageType     = vk::ImageType::e2D,
			 .format        = vk::Format::eD16Unorm,
			 .extent        = {static_cast<u32>(width), static_cast<u32>(height), 1},
			 .mipLevels     = 1,
			 .arrayLayers   = 1,
			 .samples       = vk::SampleCountFlagBits::e1,
			 .tiling        = vk::ImageTiling::eOptimal,
			 .usage         = vk::ImageUsageFlagBits::eDepthStencilAttachment,
			 .sharingMode   = vk::SharingMode::eExclusive,
			 .initialLayout = vk::ImageLayout::eUndefined,
		 },
		 .aspect = vk::ImageAspectFlagBits::eDepth});

	// After depth, because depth format is need in rendering info
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

	if (verbose) {
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

BRDFSample::~BRDFSample() { Destroy(); }

void BRDFSample::Destroy() {

	if (device) {
		CHECK_VULKAN_RESULT(device.waitIdle());

		if (timestamp_query_pool) {
			device.destroyQueryPool(timestamp_query_pool, GetAllocator());
			timestamp_query_pool = nullptr;
		}

		staging_buffer.Destroy();
		device_buffer.Destroy();

		depth_image.Destroy();
		// device_buffer.Destroy();
		// brdf_weights_buffer.Destroy();

		for (vk::Pipeline& pipeline : pipelines_header) {
			if (pipeline) {
				device.destroyPipeline(pipeline, GetAllocator());
				pipeline = vk::Pipeline{};
			}
		}
		for (vk::Pipeline& pipeline : pipelines) {
			if (pipeline) {
				device.destroyPipeline(pipeline, GetAllocator());
				pipeline = vk::Pipeline{};
			}
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

void BRDFSample::CreateInstance() {
	vk::ApplicationInfo applicationInfo{.apiVersion = kApiVersion};

	u32 glfw_extensions_count;

	char const** glfw_extensions = WindowManager::GetRequiredInstanceExtensions(&glfw_extensions_count);

	std::vector<char const*> enabledExtensions(glfw_extensions, glfw_extensions + glfw_extensions_count);
	if (use_validation) {
		enabled_layers = kEnabledLayers;
		enabledExtensions.push_back(vk::EXTDebugUtilsExtensionName);
	}

	vk::DebugUtilsMessengerCreateInfoEXT constexpr kDebugUtilsCreateInfo = {
		.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
						   //    vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
						   //    vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo |
						   vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
		.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
					   //    vk::DebugUtilsMessageTypeFlagBitsEXT::eDeviceAddressBinding |
					   vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
		.pfnUserCallback = DebugUtilsCallback,
		.pUserData       = nullptr,
	};

	vk::InstanceCreateInfo info{
		.pNext                   = use_validation ? &kDebugUtilsCreateInfo : nullptr,
		.pApplicationInfo        = &applicationInfo,
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
		.enabledExtensionCount   = static_cast<u32>(std::size(enabledExtensions)),
		.ppEnabledExtensionNames = enabledExtensions.data(),
	};
	CHECK_VULKAN_RESULT(vk::createInstance(&info, GetAllocator(), &instance));

	if (use_validation) {
		LoadInstanceDebugUtilsFunctionsEXT(instance);
		CHECK_VULKAN_RESULT(instance.createDebugUtilsMessengerEXT(&kDebugUtilsCreateInfo, allocator, &debug_messenger));
	}

	vk::Result result;
	std::tie(result, vulkan_physical_devices) = instance.enumeratePhysicalDevices();
	CHECK_VULKAN_RESULT(result);
}

void BRDFSample::SelectPhysicalDevice() {
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_VULKAN_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface, kEnabledDeviceExtensions)) {
			if (verbose) {
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

void BRDFSample::GetPhysicalDeviceInfo() {
}

void BRDFSample::CreateDevice() {
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
		.enabledExtensionCount   = static_cast<u32>(std::size(kEnabledDeviceExtensions)),
		.ppEnabledExtensionNames = kEnabledDeviceExtensions,
	};

	CHECK_VULKAN_RESULT(physical_device.createDevice(&info, GetAllocator(), &device));
	queue = device.getQueue(queue_create_infos[0].queueFamilyIndex, 0);
}

void BRDFSample::CreateVmaAllocator() {
	VmaVulkanFunctions vulkan_functions = {
		.vkGetInstanceProcAddr = &vkGetInstanceProcAddr,
		.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr,
	};

	using ::VmaAllocatorCreateFlagBits;
	VmaAllocatorCreateInfo info = {
		.flags            = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT | VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
		.physicalDevice   = physical_device,
		.device           = device,
		.pVulkanFunctions = &vulkan_functions,
		.instance         = instance,
		.vulkanApiVersion = kApiVersion,
	};
	CHECK_VULKAN_RESULT(vk::Result(vmaCreateAllocator(&info, &vma_allocator)));
}

constexpr u32 kStorageBuffersCount = 1;

void BRDFSample::CreateDescriptorSetLayout() {
	vk::DescriptorSetLayoutBinding descriptor_set_layout_binding{
		.binding         = 0,
		.descriptorType  = vk::DescriptorType::eStorageBuffer,
		.descriptorCount = kStorageBuffersCount,
		.stageFlags      = vk::ShaderStageFlagBits::eFragment,
	};

	vk::DescriptorSetLayoutCreateInfo info{
		.flags        = {},
		.bindingCount = 1,
		.pBindings    = &descriptor_set_layout_binding,
	};

	CHECK_VULKAN_RESULT(device.createDescriptorSetLayout(&info, GetAllocator(), &descriptor_set_layout));
}

void BRDFSample::CreateDescriptorPool() {
	vk::DescriptorPoolSize descriptor_pool_sizes[] = {
		{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = kStorageBuffersCount},
	};

	vk::DescriptorPoolCreateInfo info{
		.flags         = {},
		.maxSets       = 1,
		.poolSizeCount = 1,
		.pPoolSizes    = descriptor_pool_sizes,
	};

	CHECK_VULKAN_RESULT(device.createDescriptorPool(&info, GetAllocator(), &descriptor_pool));
}

void BRDFSample::CreateDescriptorSet() {
	vk::DescriptorSetAllocateInfo info{
		.descriptorPool     = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &descriptor_set_layout,
	};

	CHECK_VULKAN_RESULT(device.allocateDescriptorSets(&info, &descriptor_set));
}

void BRDFSample::CreateSwapchain() {
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

void BRDFSample::CreatePipelineLayout() {
	vk::PushConstantRange push_constant_range{
		.stageFlags = vk::ShaderStageFlagBits::eVertex
					  | vk::ShaderStageFlagBits::eFragment,
		.offset = 0,
		.size   = physical_device.GetMaxPushConstantsSize(),
	};

	vk::PipelineLayoutCreateInfo info{
		.setLayoutCount         = 1,
		.pSetLayouts            = &descriptor_set_layout,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges    = &push_constant_range,
	};

	CHECK_VULKAN_RESULT(device.createPipelineLayout(&info, GetAllocator(), &pipeline_layout));
}

void BRDFSample::CreatePipelines() {

	auto error_read_file = [] {
		std::printf("Failed to read shader file!\n");
		std::exit(1);
		return std::optional<std::vector<std::byte>>{};
	};

	std::optional<std::vector<std::byte>> shader_codes_main[] = {
		Utils::ReadBinaryFile("Shaders/BRDFMain.slang.spv").or_else(error_read_file),
	};

	std::optional<std::vector<std::byte>> shader_codes[] = {
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_2_3_65_89.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_2_4_85_117.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_2_6_125_173.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_2_8_165_229.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_2_12_245_341.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_3_3_93_120.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_3_4_122_158.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_3_6_180_234.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_3_8_238_310.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_3_12_354_462.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_4_3_121_151.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_4_4_159_199.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_4_6_235_295.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_4_8_311_391.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_4_12_463_583.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_5_3_149_182.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_5_4_196_240.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_5_6_290_356.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_5_8_384_472.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_5_12_572_704.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_6_3_177_213.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_6_4_233_281.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_6_6_345_417.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_6_8_457_553.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_1_6_12_681_825.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_2_3_82_112.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_2_4_107_147.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_2_6_157_217.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_2_8_207_287.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_2_12_307_427.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_3_3_126_162.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_3_4_165_213.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_3_6_243_315.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_3_8_321_417.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_3_12_477_621.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_4_3_176_218.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_4_4_231_287.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_4_6_341_425.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_4_8_451_563.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_4_12_671_839.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_5_3_232_280.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_5_4_305_369.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_5_6_451_547.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_5_8_597_725.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_5_12_889_1081.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_6_3_294_348.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_6_4_387_459.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_6_6_573_681.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_6_8_759_903.slang.spv").or_else(error_read_file),
		Utils::ReadBinaryFile("Shaders/SINEKAN_2_6_12_1131_1347.slang.spv").or_else(error_read_file),
		// Utils::ReadBinaryFile("Shaders/BRDFMain.slang.spv").or_else(error_read_file),
	};

	vk::ShaderModuleCreateInfo shader_module_infos[std::size(shader_codes)];
	vk::ShaderModule           shader_modules[std::size(shader_codes)];

	for (u32 i = 0; i < std::size(shader_codes); ++i) {
		shader_module_infos[i].codeSize = (*shader_codes[i]).size();
		shader_module_infos[i].pCode    = reinterpret_cast<const u32*>((*shader_codes[i]).data());
		CHECK_VULKAN_RESULT(device.createShaderModule(&shader_module_infos[i], GetAllocator(), &shader_modules[i]));
	}
	vk::ShaderModuleCreateInfo shader_module_info_main;
	vk::ShaderModule           shader_module_main;
	shader_module_info_main.codeSize = ((*shader_codes_main[0]).size());
	shader_module_info_main.pCode    = reinterpret_cast<const u32*>((*shader_codes_main[0]).data());
	CHECK_VULKAN_RESULT(device.createShaderModule(&shader_module_info_main, GetAllocator(), &shader_module_main));

	for (auto i = 0u; i < std::size(pipelines); ++i) {
		pipelines[i] = CreatePipeline(shader_module_main, {.function_type = static_cast<BrdfFunctionType>(i)});
	}

	for (auto i = 0u; i < kTestFunctionsCount; ++i) {
		pipelines_header[i] = CreatePipeline(shader_modules[i], {.function_type = function_type, .function_id = i});
	}

	for (auto& shader_module : shader_modules) {
		device.destroyShaderModule(shader_module, GetAllocator());
	}
}

auto BRDFSample::CreatePipeline(vk::ShaderModule shader_module, SpecData const& info) -> vk::Pipeline {

	// Specialization constant for type of inferencing function
	// BrdfFunctionType specialization_value = info.function_type;

	vk::SpecializationMapEntry specialization_entries[] = {
		{
			.constantID = 0,
			.offset     = offsetof(SpecData, function_type),
			.size       = sizeof(info.function_type),
		},
		{
			.constantID = 1,
			.offset     = offsetof(SpecData, function_id),
			.size       = sizeof(info.function_id),
		},
	};
	vk::SpecializationInfo specialization_info{
		.mapEntryCount = std::size(specialization_entries),
		.pMapEntries   = specialization_entries,
		.dataSize      = sizeof(info),
		.pData         = &info,
	};

	vk::PipelineShaderStageCreateInfo shader_stages[] = {
		{.stage = vk::ShaderStageFlagBits::eVertex, .module = shader_module, .pName = "vs_main"},
		{.stage = vk::ShaderStageFlagBits::eFragment, .module = shader_module, .pName = "ps_main", .pSpecializationInfo = &specialization_info},
	};

	vk::VertexInputBindingDescription vertex_input_binding_descriptions[] = {{
		.binding   = 0,
		.stride    = sizeof(Vertex),
		.inputRate = vk::VertexInputRate::eVertex,
	}};

	vk::VertexInputAttributeDescription vertex_input_attribute_descriptions[] = {
		{.location = 0, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(Vertex, pos)},
		{.location = 1, .binding = 0, .format = vk::Format::eR32G32B32A32Sfloat, .offset = offsetof(Vertex, normal)},
	};

	vk::PipelineVertexInputStateCreateInfo vertex_input_state{
		.vertexBindingDescriptionCount   = std::size(vertex_input_binding_descriptions),
		.pVertexBindingDescriptions      = vertex_input_binding_descriptions,
		.vertexAttributeDescriptionCount = std::size(vertex_input_attribute_descriptions),
		.pVertexAttributeDescriptions    = vertex_input_attribute_descriptions,
	};

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
		.cullMode  = vk::CullModeFlagBits::eBack,
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

	auto color_write_mask =
		vk::ColorComponentFlagBits::eR
		| vk::ColorComponentFlagBits::eG
		| vk::ColorComponentFlagBits::eB
		| vk::ColorComponentFlagBits::eA;

	vk::PipelineColorBlendAttachmentState color_blend_attachment_state{
		.blendEnable    = vk::False,
		.colorWriteMask = color_write_mask,
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
		.depthAttachmentFormat   = depth_image.GetFormat(),
	};

	vk::GraphicsPipelineCreateInfo create_info{
		.pNext               = &pipeline_rendering_info,
		.stageCount          = static_cast<u32>(std::size(shader_stages)),
		.pStages             = shader_stages,
		.pVertexInputState   = &vertex_input_state,
		.pInputAssemblyState = &input_assembly_state,
		.pViewportState      = &viewport_state,
		.pRasterizationState = &rasterization_state,
		.pMultisampleState   = &multisample_state,
		.pDepthStencilState  = &depth_stencil_state,
		.pColorBlendState    = &color_blend_state,
		.pDynamicState       = &dynamic_state,
		.layout              = pipeline_layout,
	};

	vk::Pipeline pipeline;
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &create_info, GetAllocator(), &pipeline));

	return pipeline;
}
