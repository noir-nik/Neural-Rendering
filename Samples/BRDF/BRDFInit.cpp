module BRDFSample;

#include "CheckResult.h"
#include "Log.h"
#include "Shaders/BRDFBindings.h"
#include "Shaders/BRDFConfig.h"

import NeuralGraphics;
import vulkan;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import WeightsLoader;
import SamplesCommon;
import Math;
import std;
#include "BRDFVulkanConstants.h"

static constexpr auto kApiVersion = vk::ApiVersion13;

static constexpr char const* kEnabledLayers[] = {
	"VK_LAYER_KHRONOS_validation",
};
static constexpr std::array kEnabledDeviceExtensionsArr = {
	vk::KHRSwapchainExtensionName,
	// vk::NVCooperativeVectorExtensionName,
	// vk::EXTShaderReplicatedCompositesExtensionName,
};
static std::span kEnabledDeviceExtensions = {kEnabledDeviceExtensionsArr.data(), kEnabledDeviceExtensionsArr.size()};

static constexpr std::array kEnabledDeviceExtensionsCoopVecArr = {
	vk::KHRSwapchainExtensionName,
	vk::NVCooperativeVectorExtensionName,
	vk::EXTShaderReplicatedCompositesExtensionName,
};
static std::span kEnabledDeviceExtensionsCoopVec = {kEnabledDeviceExtensionsCoopVecArr.data(), kEnabledDeviceExtensionsCoopVecArr.size()};

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

#if defined(WITH_UI) && WITH_UI
		case Key::eE: {
			sample->is_ui_visible = !sample->is_ui_visible;
		} break;
		case Key::eF: {
			sample->is_fps_visible = !sample->is_fps_visible;
		} break;
		case Key::eD: {
			sample->is_models_visible = !sample->is_models_visible;
		} break;
		case Key::eC: {
			sample->set_cubemap_visible(!sample->is_cubemap_visible_);
		} break;
#endif
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
	LOG_DEBUG("BRDFSample::Init()");
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();
	u32 const initial_width = 1600, initial_height = 1200;
	// u32 const initial_width = 1920, initial_height = 1080;

	window.Init({
		.x          = 30,
		.y          = 30,
		.width      = initial_width,
		.height     = initial_height,
		.title      = "BRDF Sample",
		.bDecorated = false,
	});

	window.SetPos(0, 0);

	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!is_test_mode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}

	int x, y, width, height;
	window.GetFullScreenRect(x, y, width, height);

	auto invec = (float3{0.0f, 0.0f, 2.2f} * 1.2);
	invec      = rotate(invec, {0, 1, 0}, -100 * math::DEG_TO_RAD);
	camera     = {{
		.position = invec,
		.fov      = 35.0f,
		.z_near   = 0.01f,
		.z_far    = 100.0f,
	}};

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

	// ENABLE_FAST_KAN
	if (0)
		if (kan_weights_file_name.size() == 0) {
			std::printf("Enter the path to the kan weights file");
			// char  buffer[256];
			// char* result      = std::fgets(buffer, sizeof(buffer), stdin);
			// kan_weights_file_name = buffer;
			std::exit(1);
		}
	// ReadKANWeights({.file_name = kan_weights_file_name, .header = ""});

	depth_image.Create(
		device, vma_allocator, allocator,
		{.image_info = {
			 .flags     = {},
			 .imageType = vk::ImageType::e2D,
			 //  .format    = vk::Format::eD16Unorm,
			 .format = vk::Format::eD32Sfloat,
			 //  .format        = vk::Format::eD24UnormS8Uint,
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

	accumulator_image.Create(
		device, vma_allocator, allocator,
		{.image_info = {
			 .flags         = {},
			 .imageType     = vk::ImageType::e2D,
			 .format        = vk::Format::eR32G32B32A32Sfloat,
			 .extent        = {static_cast<u32>(width), static_cast<u32>(height), 1},
			 .mipLevels     = 1,
			 .arrayLayers   = 1,
			 .samples       = vk::SampleCountFlagBits::e1,
			 .tiling        = vk::ImageTiling::eOptimal,
			 .usage         = vk::ImageUsageFlagBits::eStorage,
			 .sharingMode   = vk::SharingMode::eExclusive,
			 .initialLayout = vk::ImageLayout::eUndefined,
		 },
		 .aspect = vk::ImageAspectFlagBits::eColor});

	auto const _file_name =
		hasattr(&BRDFSample::weights_file_name)
			? weights_file_name
			: "Assets/simple_brdf_weights.bin";

	// auto header =
	// 	//
	// 	""
	// 	//  "hydrann1"
	// 	;
	CreateAndUploadBuffers({.file_name = _file_name, .header = this->header});

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

	if (with_coop_vec() && verbose) {
		auto [result, cooperative_vector_properties] = physical_device.getCooperativeVectorPropertiesNV();
		CHECK_VULKAN_RESULT(result);
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

#if defined(WITH_UI) && WITH_UI
	UI::Init();
	{
		vk::DescriptorPoolSize imguiPoolSizes[] = {
			{vk::DescriptorType::eUniformBuffer, 1000},
			{vk::DescriptorType::eCombinedImageSampler, 1000},
		};
		vk::DescriptorPoolCreateInfo info{
			.flags         = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
			.maxSets       = (u32)(1024),
			.poolSizeCount = std::size(imguiPoolSizes),
			.pPoolSizes    = imguiPoolSizes,
		};
		CHECK_VULKAN_RESULT(device.createDescriptorPool(&info, GetAllocator(), &imgui_descriptor_pool));
	}
	CreateImGui(); // <- imgui_descriptor_pool
#endif
}

BRDFSample::~BRDFSample() {
	LOG_DEBUG("BRDFSample::BRDFSample()");
	Destroy();
}

void BRDFSample::Destroy() {
	LOG_DEBUG("BRDFSample::Destroy()");

	if (device) {
		CHECK_VULKAN_RESULT(device.waitIdle());

#if defined(WITH_UI) && WITH_UI
		ImGuiShutdown();
		UI::Destroy();
		device.destroyDescriptorPool(imgui_descriptor_pool, GetAllocator());
#endif

		if (timestamp_query_pool) {
			device.destroyQueryPool(timestamp_query_pool, GetAllocator());
			timestamp_query_pool = nullptr;
		}

		staging_buffer.Destroy();
		device_buffer.Destroy();

		depth_image.Destroy();
		// for (auto& im : cubemap_images) {
		// }
		cubemap_image.Destroy();
		accumulator_image.Destroy();

		if (cubemap_sampler) {
			device.destroySampler(cubemap_sampler, GetAllocator());
			cubemap_sampler = nullptr;
		}
		// device_buffer.Destroy();
		// brdf_weights_buffer.Destroy();

		auto mkspan        = [](auto& rng) { return std::span(rng.begin(), rng.end()); };
		auto mkspan_single = [](auto& elem) { return std::span(&elem, 1); };

		for (auto pp_span : {
				 mkspan(pipelines),
				 mkspan(generated_pipelines),
				 mkspan(pipelines_fallback),
				 mkspan_single(skybox_pipeline),
			 }) {
			for (vk::Pipeline& pipeline : pp_span) {
				if (pipeline) {
					device.destroy(pipeline, GetAllocator());
					pipeline = vk::Pipeline{};
				}
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
	LOG_DEBUG("BRDFSample::CreateInstance()");
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
		.messageType     = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
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
	LOG_DEBUG("BRDFSample::SelectPhysicalDevice()");
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_VULKAN_RESULT(physical_device.GetDetails());
		// auto const arr = ;

		if (physical_device.IsSuitable(surface, {with_coop_vec() ? kEnabledDeviceExtensionsCoopVec : kEnabledDeviceExtensions})) {
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
	LOG_DEBUG("BRDFSample::GetPhysicalDeviceInfo()");
}

void BRDFSample::CreateDevice() {
	LOG_DEBUG("BRDFSample::CreateDevice()");
	float const queue_priorities[] = {1.0f};

	auto [result, index] = physical_device.GetQueueFamilyIndex({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
	if (result != vk::Result::eSuccess || !index.has_value()) {
		std::printf("Failed to get graphics queue family index with surface support\n");
		CHECK_VULKAN_RESULT(result);
	}

	queue_family_index = index.value();

	auto queue_family_properties = physical_device.GetQueueFamilyProperties(queue_family_index);
	timestamps_supported         = queue_family_properties.timestampValidBits > 0;

	// auto timestamp_period = physical_device.GetProperties10().limits.timestampPeriod;
	// std::printf("timestampPeriod: %f\n", timestamp_period);

	vk::DeviceQueueCreateInfo queue_create_infos[] = {
		{
			.queueFamilyIndex = queue_family_index,
			.queueCount       = 1,
			.pQueuePriorities = queue_priorities,
		}};

	auto physicalDeviceShaderReplicatedCompositesFeaturesEXT = vk::PhysicalDeviceShaderReplicatedCompositesFeaturesEXT{
		.shaderReplicatedComposites = vk::True,
	};
	auto physicalDeviceCooperativeVectorFeaturesNV = vk::PhysicalDeviceCooperativeVectorFeaturesNV{
		.pNext                     = &physicalDeviceShaderReplicatedCompositesFeaturesEXT,
		.cooperativeVector         = vk::True,
		.cooperativeVectorTraining = vk::True,
	};

	auto physicalDeviceVulkan13Features = vk::PhysicalDeviceVulkan13Features{
		.pNext            = with_coop_vec() ? &physicalDeviceShaderReplicatedCompositesFeaturesEXT : 0,
		.synchronization2 = vk::True,
		.dynamicRendering = vk::True,
	};
	auto physicalDeviceVulkan12Features = vk::PhysicalDeviceVulkan12Features{
		.pNext               = &physicalDeviceVulkan13Features,
		.shaderFloat16       = vk::True,
		.hostQueryReset      = timestamps_supported ? vk::True : vk::False,
		.bufferDeviceAddress = vk::True,
	};
	auto physicalDeviceVulkan11Features = vk::PhysicalDeviceVulkan11Features{
		.pNext                    = &physicalDeviceVulkan12Features,
		.storageBuffer16BitAccess = vk::True,
		.shaderDrawParameters     = vk::True,
	};

	auto physicalDeviceFeatures2 = vk::PhysicalDeviceFeatures2{
		.pNext    = &physicalDeviceVulkan11Features,
		.features = vk::PhysicalDeviceFeatures{
			.samplerAnisotropy        = physical_device.GetFeatures2().features.samplerAnisotropy,
			.fragmentStoresAndAtomics = physical_device.GetFeatures2().features.fragmentStoresAndAtomics,
		},
	};

	auto& features = physicalDeviceFeatures2;

	auto                 ars = with_coop_vec() ? kEnabledDeviceExtensionsCoopVec : kEnabledDeviceExtensions;
	vk::DeviceCreateInfo info{
		// .pNext                   = &features.get<vk::PhysicalDeviceFeatures2>(),
		.pNext                   = &features,
		.queueCreateInfoCount    = static_cast<u32>(std::size(queue_create_infos)),
		.pQueueCreateInfos       = queue_create_infos,
		.enabledExtensionCount   = static_cast<u32>(std::size(ars)),
		.ppEnabledExtensionNames = std::data(ars),
	};

	CHECK_VULKAN_RESULT(physical_device.createDevice(&info, GetAllocator(), &device));
	queue = device.getQueue(queue_create_infos[0].queueFamilyIndex, 0);
}

void BRDFSample::CreateVmaAllocator() {
	LOG_DEBUG("BRDFSample::CreateVmaAllocator()");
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

constexpr u32 kStorageBuffersCount      = 1;
constexpr u32 CombinedImageSamplerCount = 1;
constexpr u32 StorageImageCount         = 1;

void BRDFSample::CreateDescriptorSetLayout() {
	LOG_DEBUG("BRDFSample::CreateDescriptorSetLayout()");
	vk::DescriptorSetLayoutBinding descriptor_set_layout_bindings[] = {
		{
			.binding         = BINDING_STORAGE_BUFFER,
			.descriptorType  = vk::DescriptorType::eStorageBuffer,
			.descriptorCount = kStorageBuffersCount,
			.stageFlags      = vk::ShaderStageFlagBits::eFragment,
		},
		{
			.binding         = BINDING_TEXTURE,
			.descriptorType  = vk::DescriptorType::eCombinedImageSampler,
			.descriptorCount = CombinedImageSamplerCount,
			.stageFlags      = vk::ShaderStageFlagBits::eFragment,
		},

		{
			.binding         = BINDING_STORAGE_IMAGE,
			.descriptorType  = vk::DescriptorType::eStorageImage,
			.descriptorCount = StorageImageCount,
			.stageFlags      = vk::ShaderStageFlagBits::eFragment,
		}};

	vk::DescriptorSetLayoutCreateInfo info{
		.flags        = {},
		.bindingCount = static_cast<decltype(info.bindingCount)>(std::size(descriptor_set_layout_bindings)),
		.pBindings    = descriptor_set_layout_bindings,
	};

	CHECK_VULKAN_RESULT(device.createDescriptorSetLayout(&info, GetAllocator(), &descriptor_set_layout));
}

void BRDFSample::CreateDescriptorPool() {
	LOG_DEBUG("BRDFSample::CreateDescriptorPool()");
	vk::DescriptorPoolSize descriptor_pool_sizes[] = {
		{.type = vk::DescriptorType::eStorageBuffer, .descriptorCount = kStorageBuffersCount},
		{.type = vk::DescriptorType::eCombinedImageSampler, .descriptorCount = CombinedImageSamplerCount},
		{.type = vk::DescriptorType::eStorageImage, .descriptorCount = StorageImageCount},
	};

	vk::DescriptorPoolCreateInfo info{
		.flags         = {},
		.maxSets       = 1,
		.poolSizeCount = static_cast<decltype(info.poolSizeCount)>(std::size(descriptor_pool_sizes)),
		.pPoolSizes    = descriptor_pool_sizes,
	};

	CHECK_VULKAN_RESULT(device.createDescriptorPool(&info, GetAllocator(), &descriptor_pool));
}

void BRDFSample::CreateDescriptorSet() {
	LOG_DEBUG("BRDFSample::CreateDescriptorSet()");
	vk::DescriptorSetAllocateInfo info{
		.descriptorPool     = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts        = &descriptor_set_layout,
	};

	CHECK_VULKAN_RESULT(device.allocateDescriptorSets(&info, &descriptor_set));
}

void BRDFSample::CreateSwapchain() {
	LOG_DEBUG("BRDFSample::CreateSwapchain()");
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
	LOG_DEBUG("BRDFSample::CreatePipelineLayout()");
	vk::PushConstantRange push_constant_range{
		.stageFlags = vk::ShaderStageFlagBits::eVertex
					  | vk::ShaderStageFlagBits::eFragment,
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

// template <typename... Args>
// auto cat(Args&&... args)
// requires(std::is_convertible_v<Args, std::string_view> && ...)
// {
// 	return Utils::StringViewCat(std::forward<Args>(args)...);
// }

// constexpr auto st = cat("Shaders/BRDFMain.slang.spv");
