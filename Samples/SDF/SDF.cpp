#include "stddef.h" // offsetof

#include "CheckResult.h"
#include "Shaders/SDFConfig.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;
import std;
import Math;

using numeric::float16_t;
using float32_t = float;
using namespace math;

extern "C++" {
#include "Shaders/SDFConstants.h"
}

#define _TY float

#include "Shaders/SDFWeights.h"
#define WEIGHTS_H
// #include "SDFWeightsInHeader_3_16_16_16_1_625.h"
// #include "SDFWeightsInHeader_3_24_24_24_1_1321.h"
// #include "SDFWeightsInHeader_3_32_32_32_1_2273.h"
// #include "SDFWeightsInHeader_3_32_32_32_32_1_3329.h"
// #include "SDFWeightsInHeader_3_48_48_48_1_4945.h"

// #include "Shaders/SDFWeightsInHeader_3_16_16_16_16_1_897.h"

using namespace Utils;
using namespace mesh;

using StrView = std::string_view;

struct TestOptions {
	int2 resolution = {640, 480};
	// TestType    test_type    = TestType::eSDF;
	// NetworkType network_type = NetworkType::eScalarInline;
	int test_count = 1;
	// StrView weights_file;
};
struct InitInfo {
	std::string_view weights_file;
};
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

	static constexpr u32 kFramesInFlight     = 3;
	static constexpr u32 kTestFunctionsCount = 8;
	static constexpr u32 kNetworksCount      = 3;
	static constexpr u32 kTimestampsPerFrame = 2;

	~SDFSample();

	void Init(InitInfo const& init_info);
	void Run();
	void RunBenchmark(TestOptions const& options);
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

	struct SpecData {
		SdfFunctionType function_type = SdfFunctionType::eCoopVec;
		u32             function_id   = 0; // inline
	};

	void CreatePipelines();
	[[nodiscard]]
	auto CreatePipeline(
		vk::ShaderModule vertex_shader_module,
		vk::ShaderModule fragment_shader_module,
		SpecData const&  spec) -> vk::Pipeline;

	struct NetworkBufferInfo {
		std::string_view file_name;
		std::string_view header;
	};
	// void BuildNetwork();
	// void CreateAndUploadBuffers();
	void CreateAndUploadBuffers(NetworkBufferInfo const& info);

	void RecordCommands(vk::Pipeline pipeline);
	// Return time in nanoseconds
	auto GetQueryResult() -> u64;
	auto DrawWindow(vk::Pipeline pipeline) -> u64;
	auto DrawWindow() -> u64 {
		return DrawWindow(pipelines[function_id.value_or(0)]);
	}
	// auto DrawWindow() -> u64 { return DrawWindow(pipelines[u32(SdfFunctionType::eScalarBuffer)], row_major_offsets); }
	void RecreateSwapchain(int width, int height);
	void SaveSwapchainImageToFile(std::string_view filename);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }
	bool IsTestMode() const { return is_test_mode; }

	auto ParseArgs(int argc, char const* argv[]) -> char const*;

	bool is_test_mode   = false;
	bool is_verbose     = false;
	bool use_validation = false;
	bool pics           = false;

	std::optional<u32> function_id = std::nullopt;

	GLFWWindow window{};
	Mouse      mouse;

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

	SdfFunctionType function_type = SdfFunctionType::eCoopVec;
	// std::string_view weights_file_name;

	// std::array<vk::Pipeline, u32(SdfFunctionType::eCount)> pipelines;
	// std::array<vk::Pipeline, u32(SdfFunctionType::eCount)> pipelines;

	// vk::Pipeline pipelines[u32(SdfFunctionType::eCount)][kTestFunctionsCount];
	// vk::Pipeline pipelines[kTestFunctionsCount];

	std::array<vk::Pipeline, kTestFunctionsCount> pipelines;

	VulkanRHI::Buffer staging_buffer{};
	VulkanRHI::Buffer device_buffer{};

	VulkanCoopVecNetwork networks[kNetworksCount];

	std::array<vk::DeviceSize, kTestFunctionsCount> weights_offsets;

	bool pending_image_save = false;

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

	Camera camera{{
		// .position = {-4.180247, -0.427392, 0.877357},
		.position = {2.7, 0.8, -4.3},
		// .position = {3.2, 1.21, -4.3},
		// .focus    = {-0.1f, 0.0f, 0.0f},
		.focus = {},
		// .focus = {-0.3f, 0.65f, -0.0f},
		// .up    = {0.213641, -0.093215, 0.972476},
		.up = {0, 1, 0},
		// .fov    = 35.0f,
		.fov = 32.0f,
		// .fov    = 10.0f,
		.z_near = 0.01f,
		.z_far  = 1000.0f,
	}};
};
namespace {
static void FramebufferSizeCallback(GLFWWindow* window, int width, int height) {
	SDFSample* sample = static_cast<SDFSample*>(window->GetUserPointer());

	sample->swapchain_dirty = true;
	if (width <= 0 || height <= 0) return;
	sample->RecreateSwapchain(width, height);
	sample->camera.updateProjection(width, height);
}

static void WindowRefreshCallback(GLFWWindow* window) {
	SDFSample* sample = static_cast<SDFSample*>(window->GetUserPointer());

	int x, y, width, height;
	window->GetRect(x, y, width, height);
	if (width <= 0 || height <= 0) return;
	sample->DrawWindow();
}

static void CursorPosCallback(GLFWWindow* window, double xpos, double ypos) {
	SDFSample* sample = static_cast<SDFSample*>(window->GetUserPointer());

	sample->mouse.delta_x = static_cast<float>(xpos - sample->mouse.x);
	sample->mouse.delta_y = -static_cast<float>(ypos - sample->mouse.y);
	sample->mouse.x       = static_cast<float>(xpos);
	sample->mouse.y       = static_cast<float>(ypos);

	ProcessViewportInput(sample->window, sample->camera, sample->mouse, sample->mouse.delta_x, sample->mouse.delta_y);
}

static void KeyCallback(GLFWWindow* window, int key, int scancode, int action, int mods) {
	SDFSample* sample = static_cast<SDFSample*>(window->GetUserPointer());
	using namespace Glfw;
	if (Action(action) == Action::ePress) {
		switch (Key(key)) {
		case Key::eEscape:
			window->SetShouldClose(true);
			break;
		case Key::eF8: {
			sample->pending_image_save = true;

			// auto fname = "sdf.bmp";

			char fname[256] = {};

			std::snprintf(fname, sizeof(fname), "sdf_%d.bmp", *sample->function_id);

			sample->SaveSwapchainImageToFile(fname);
			// sample->SaveSwapchainImageToFile("sdf.png");
		} break;
		default:
			break;
		}
	}
}

static void MouseButtonCallback(GLFWWindow* in_window, int in_button, int in_action, int in_mods) {
	SDFSample* sample = static_cast<SDFSample*>(in_window->GetUserPointer());
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

void SDFSample::Init(InitInfo const& init_info) {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();

	// u32 const initial_width = 1600, initial_height = 1200;
	u32 const initial_width = 800, initial_height = 600;
	// u32 const initial_width = 1920, initial_height = 1080;
	window.Init({.x = 30, .y = 30, .width = initial_width, .height = initial_height, .title = "SDF"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!is_test_mode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}
	
	if (false) {
		camera.getPosition() = {3.2, 1.21, -4.3};
		camera.getFocus()    = {-0.3f, 0.65f, -0.0f};
		camera.getFov()      = 10.0f;
	}

	int x, y, width, height;
	window.GetFullScreenRect(x, y, width, height);

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
	CreateAndUploadBuffers({.file_name = init_info.weights_file.size() > 0 ? init_info.weights_file : "Assets/simple_brdf_weights.bin", .header = ""});

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

	if (is_verbose) {
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
		device_buffer.Destroy();

		// for (auto& pip : pipelines) {
		for (auto& pipeline : pipelines) {
			// for (auto& pipeline : pip) {
			device.destroyPipeline(pipeline, GetAllocator());
			pipeline = vk::Pipeline{};
		}
		// }
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

void SDFSample::SelectPhysicalDevice() {
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_VULKAN_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface, kEnabledDeviceExtensions)) {
			if (is_verbose) {
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
			.vulkanMemoryModel   = vk::True,
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
		.flags            = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT | VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT | VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
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

	for (auto const& code : shader_codes) {
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

	// pipelines[int(SdfFunctionType::eCoopVec)]         = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eCoopVec);
	// pipelines[int(SdfFunctionType::eWeightsInHeader)] = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eWeightsInHeader);
	// pipelines[int(SdfFunctionType::eWeightsInBuffer)] = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eWeightsInBuffer);
	// pipelines[int(SdfFunctionType::eVec4)]            = CreatePipeline(shader_modules[0], shader_modules[1], SdfFunctionType::eVec4);

	// u32 num = 6u;
	u32 num = 7u;
	// u32 num = 8u;
	// for (auto i = num; i < num + 1; ++i) {
	for (auto i = 0u; i < std::size(pipelines); ++i) {
		if (is_verbose) {
			std::printf("Creating pipeline %d\n", i);
		}
		pipelines[i] = CreatePipeline(shader_modules[0], shader_modules[1], {function_type, i});
	}

	for (auto& shader_module : shader_modules) {
		device.destroyShaderModule(shader_module, GetAllocator());
	}
}

auto SDFSample::CreatePipeline(vk::ShaderModule vertex_shader_module,
							   vk::ShaderModule fragment_shader_module,
							   SpecData const&  info) -> vk::Pipeline {

	// Specialization constant for type of inferencing function
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
		.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
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

	vk::GraphicsPipelineCreateInfo create_info{
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
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &create_info, GetAllocator(), &pipeline));

	return pipeline;
}

void SDFSample::CreateAndUploadBuffers(NetworkBufferInfo const& network_info) {
	// sphere = UVSphere(1.0f, 32*2, 16*2);
	// std::size_t vertices_size_bytes   = sphere.GetVertexCount() * sizeof(UVSphere::Vertex);
	// std::size_t indices_size_bytes    = sphere.GetIndexCount() * sizeof(UVSphere::IndexType);
	std::size_t alignment = sizeof(float) * 4;
	// std::size_t vertices_size_aligned = AlignUpPowerOfTwo(vertices_size_bytes, alignment);
	// std::size_t indices_size_aligned  = AlignUpPowerOfTwo(indices_size_bytes, alignment);

	// auto weights_path = "Assets/simple_brdf_weights.bin";

	std::vector<float>        brdf_weights_vec;
	std::vector<LayerVariant> layers;
	CHECK(load_weights(network_info.file_name.data(), layers, brdf_weights_vec, network_info.header.data()));

	// for (auto const& layer : layers) {
	// 	std::printf("Layer %u-> %u\n", layer.GetInputsCount(), layer.GetOutputsCount());
	// }

	using Component = vk::ComponentTypeKHR;
	using Layout    = vk::CooperativeVectorMatrixLayoutNV;
	// if (layers.size() != expected_layer_count) {
	// 	std::printf("Error loading weights : wrong number of layers\n");
	// 	std::exit(1);
	// }
	networks[u32(SdfFunctionType::eWeightsInBuffer)].Init(layers);
	networks[u32(SdfFunctionType::eWeightsInBufferF16)].Init(layers);
	networks[u32(SdfFunctionType::eCoopVec)].Init(layers);

	CHECK_VULKAN_RESULT(networks[u32(SdfFunctionType::eCoopVec)].UpdateOffsetsAndSize(device, Layout::eInferencingOptimal, Component::eFloat16, Component::eFloat16));
	CHECK_VULKAN_RESULT(networks[u32(SdfFunctionType::eWeightsInBuffer)].UpdateOffsetsAndSize(device, Layout::eRowMajor, Component::eFloat32, Component::eFloat32));
	CHECK_VULKAN_RESULT(networks[u32(SdfFunctionType::eWeightsInBufferF16)].UpdateOffsetsAndSize(device, Layout::eRowMajor, Component::eFloat16, Component::eFloat16));

	vk::DeviceSize const kMinSize = 8 * 1024 * 1024;

	std::size_t total_size_bytes = std::max(
		kMinSize,
		0
			+ networks[u32(SdfFunctionType::eCoopVec)].GetParametersSize()
			+ networks[u32(SdfFunctionType::eWeightsInBuffer)].GetParametersSize()
			+ networks[u32(SdfFunctionType::eWeightsInBufferF16)].GetParametersSize());

	// clang-format off
	CHECK_VULKAN_RESULT(device_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferDst 
				| vk::BufferUsageFlagBits::eStorageBuffer
				| vk::BufferUsageFlagBits::eVertexBuffer
				| vk::BufferUsageFlagBits::eIndexBuffer,
		.memory = vk::MemoryPropertyFlagBits::eDeviceLocal,
	}));
 
	CHECK_VULKAN_RESULT(staging_buffer.Create(device, vma_allocator, {
		.size   = total_size_bytes,
		.usage  = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible 
				| vk::MemoryPropertyFlagBits::eHostCoherent,
	}));
	// clang-format on

	// Update descriptor set
	vk::DescriptorBufferInfo buffer_infos[] = {{.buffer = device_buffer, .offset = 0, .range = device_buffer.GetSize()}};

	vk::WriteDescriptorSet writes[] = {{
		.dstSet          = descriptor_set,
		.dstBinding      = 0,
		.dstArrayElement = 0,
		.descriptorCount = static_cast<u32>(std::size(buffer_infos)),
		.descriptorType  = vk::DescriptorType::eStorageBuffer,
		.pBufferInfo     = buffer_infos,
	}};

	auto descriptor_copy_count = 0u;
	auto copy_descriptor_sets  = nullptr;
	device.updateDescriptorSets(std::size(writes), writes, descriptor_copy_count, copy_descriptor_sets);

	// this->vertices_offset = 0;
	// this->indices_offset  = this->vertices_offset + vertices_size_aligned;

	// this->linear_weights_offset     = this->indices_offset + AlignUpPowerOfTwo(indices_size_bytes, CoopVecUtils::GetMatrixAlignment());
	// this->linear_weights_offset_f16 = this->linear_weights_offset + AlignUpPowerOfTwo(networks[u32(SdfFunctionType::eScalarBuffer)].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	// this->optimal_weights_offset    = this->linear_weights_offset_f16 + AlignUpPowerOfTwo(networks[u32(SdfFunctionType::eScalarBufferF16)].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());

	auto offset = 0;

	auto kGap = 128 * 1024;
	for (auto i = 0u; i < std::size(networks); ++i) {
		offset             = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());
		weights_offsets[i] = offset;
		offset += AlignUpPowerOfTwo(networks[i].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
		// offset += kGap;
	}

	auto p_staging = static_cast<std::byte*>(staging_buffer.GetMappedData());

	auto write_weights = [&](
							 void const*                         src,
							 std::size_t                         src_size,
							 std::byte*                          dst,
							 vk::CooperativeVectorMatrixLayoutNV dst_layout,
							 vk::ComponentTypeKHR                src_component_type,
							 vk::ComponentTypeKHR                dst_matrix_type,
							 Linear const&                       linear) {
		std::size_t expected_size = linear.GetWeightsSize();
		std::size_t required_size = expected_size;

		vk::ConvertCooperativeVectorMatrixInfoNV info{
			.srcSize          = src_size,
			.srcData          = {.hostAddress = src},
			.pDstSize         = &required_size,
			.dstData          = {.hostAddress = dst + linear.GetWeightsOffset()},
			.srcComponentType = src_component_type,
			.dstComponentType = dst_matrix_type,
			.numRows          = linear.GetOutputsCount(),
			.numColumns       = linear.GetInputsCount(),
			.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
			.srcStride        = linear.GetInputsCount() * GetVulkanComponentSize(src_component_type),
			.dstLayout        = dst_layout,
			.dstStride        = linear.GetInputsCount() * GetVulkanComponentSize(dst_matrix_type),
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

	auto brdf_weights_src = std::span{reinterpret_cast<std::byte*>(brdf_weights_vec.data()), brdf_weights_vec.size() * sizeof(float)};

	auto write_network = [write_weights]<typename SrcBiasType, typename DstBiasType>(
							 VulkanCoopVecNetwork const&         network,
							 std::byte*                          src_parameters,
							 std::byte*                          dst_parameters,
							 vk::CooperativeVectorMatrixLayoutNV dst_layout,
							 vk::ComponentTypeKHR                src_component_type,
							 vk::ComponentTypeKHR                dst_matrix_type) {
		auto src_offset = std::size_t{0};
		for (u32 i = 0; i < network.GetLayers().size(); ++i) {
			auto& layer = network.GetLayer<Linear>(i);

			std::size_t const src_weights_size_bytes = layer.GetWeightsCount() * GetVulkanComponentSize(src_component_type);
			std::size_t const src_biases_size_bytes  = layer.GetBiasesCount() * GetVulkanComponentSize(src_component_type);

			auto const* src_weights = src_parameters + src_offset;
			write_weights(src_weights, src_weights_size_bytes, dst_parameters, dst_layout, src_component_type, dst_matrix_type, layer);
			src_offset += src_weights_size_bytes;

			auto const* src_bias = src_parameters + src_offset;
			std::byte*  dst_bias = dst_parameters + layer.GetBiasesOffset();
			// std::memcpy(dst_bias, src_bias, src_biases_size_bytes);
			for (u32 j = 0; j < layer.GetBiasesCount(); ++j) {
				DstBiasType* p_dst = reinterpret_cast<DstBiasType*>(dst_bias + j * sizeof(DstBiasType));
				*p_dst             = static_cast<DstBiasType>(reinterpret_cast<SrcBiasType const*>(src_bias)[j]);
			}
			src_offset += src_biases_size_bytes;
		}
	};

	write_network.template operator()<float, numeric::float16_t>(networks[u32(SdfFunctionType::eCoopVec)], brdf_weights_src.data(), p_staging + weights_offsets[u32(SdfFunctionType::eCoopVec)], Layout::eInferencingOptimal, Component::eFloat32, Component::eFloat16);
	write_network.template operator()<float, numeric::float16_t>(networks[u32(SdfFunctionType::eWeightsInBufferF16)], brdf_weights_src.data(), p_staging + weights_offsets[u32(SdfFunctionType::eWeightsInBufferF16)], Layout::eRowMajor, Component::eFloat32, Component::eFloat16);
	write_network.template operator()<float, float>(networks[u32(SdfFunctionType::eWeightsInBuffer)], brdf_weights_src.data(), p_staging + weights_offsets[u32(SdfFunctionType::eWeightsInBuffer)], Layout::eRowMajor, Component::eFloat32, Component::eFloat32);

	// networks[u32(SdfFunctionType::eCoopVec)].Print();
	// networks[u32(SdfFunctionType::eWeightsInBufferF16)].Print();
	// networks[u32(SdfFunctionType::eWeightsInBuffer)].Print();

	// DumpVertexData({p_vertices, sphere.GetVertexCount()}, {p_indices, sphere.GetIndexCount()});

	vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	vk::BufferCopy regions[] = {

		// Linear
		{
			.srcOffset = weights_offsets[u32(SdfFunctionType::eWeightsInBuffer)],
			.dstOffset = weights_offsets[u32(SdfFunctionType::eWeightsInBuffer)],
			.size      = networks[u32(SdfFunctionType::eWeightsInBuffer)].GetParametersSize(),
		},
		// Linear f16
		{
			.srcOffset = weights_offsets[u32(SdfFunctionType::eWeightsInBufferF16)],
			.dstOffset = weights_offsets[u32(SdfFunctionType::eWeightsInBufferF16)],
			.size      = networks[u32(SdfFunctionType::eWeightsInBufferF16)].GetParametersSize(),
		},
		// Optimal
		{
			.srcOffset = weights_offsets[u32(SdfFunctionType::eCoopVec)],
			.dstOffset = weights_offsets[u32(SdfFunctionType::eCoopVec)],
			.size      = networks[u32(SdfFunctionType::eCoopVec)].GetParametersSize(),
		},
	};

	cmd.copyBuffer(staging_buffer, device_buffer, std::size(regions), regions);

	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit({{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_VULKAN_RESULT(queue.waitIdle());
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

auto SDFSample::DrawWindow(vk::Pipeline pipeline) -> u64 {
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
	RecordCommands(pipeline);
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;

	// Call it here
	u64 elapsed = GetQueryResult();
	swapchain.EndFrame();
	return elapsed;
}

void SDFSample::RecordCommands(vk::Pipeline pipeline) {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	vk::Rect2D               render_rect{{0, 0}, {static_cast<u32>(width), static_cast<u32>(height)}};
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
			.loadOp      = vk::AttachmentLoadOp::eClear,
			.storeOp     = vk::AttachmentStoreOp::eStore,
			.clearValue  = {{{{0.1f, 0.1f, 0.1f, 1.0f}}}},
		}}},
	});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	// SDFConstants constants{
	// 	.resolution = {static_cast<float>(width), static_cast<float>(height)},
	// 	.mouse      = {mouse.x, static_cast<float>(height) - mouse.y},
	// };

	auto PrintMat4 = [](float4x4 const& mat) {
		for (int i = 0; i < 4; ++i) {
			std::printf("%f %f %f %f\n", mat[i][0], mat[i][1], mat[i][2], mat[i][3]);
		}
	};

	camera.updateProjectionViewInverse();
	SDFConstants constants{
		.camera_pos     = camera.getPosition(),
		.resolution_x   = static_cast<float>(width),
		.camera_forward = camera.getForward(),
		.resolution_y   = static_cast<float>(height),
		.camera_up      = camera.getUp(),
		.fov            = camera.getFov(),
		.camera_right   = camera.getRight(),
	};
	// PrintMat4(camera.getView());

	// auto offsets = GetOffsets();

	// for (auto i = 0u; i < network.GetLayers().size(); ++i) {

	// 	constants.weights_offsets[i] = offsets.weights_offsets[i];
	// 	constants.bias_offsets[i]    = offsets.biases_offsets[i];
	// }
	if (u32(function_type) < kNetworksCount) {
		auto const& network = networks[u32(function_type)];
		for (int i = 0; i < network.GetLayers().size(); ++i) {
			auto layer                   = network.GetLayer<Linear>(i);
			auto offset_base             = weights_offsets[u32(function_type)];
			constants.weights_offsets[i] = offset_base + layer.GetWeightsOffset();
			constants.bias_offsets[i]    = offset_base + layer.GetBiasesOffset();
			// std::printf("Layer %d weights_offset %d bias_offset %d\n", i, constants.weights_offsets[i], constants.bias_offsets[i]);
		}
	}
	// std::printf("W offsets: %u %u %u %u\n", offsets.weights_offsets[0], offsets.weights_offsets[1], offsets.weights_offsets[2], offsets.weights_offsets[3]);
	// std::printf("B offsets: %u %u %u %u\n", offsets.biases_offsets[0], offsets.biases_offsets[1], offsets.biases_offsets[2], offsets.biases_offsets[3]);

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

void SDFSample::SaveSwapchainImageToFile(std::string_view filename) {
	CHECK_VULKAN_RESULT(device.waitForFences(1, &swapchain.GetCurrentFence(), vk::True, std::numeric_limits<u32>::max()));

	int x, y, width, height;
	window.GetRect(x, y, width, height);
	auto const image             = swapchain.GetCurrentImage();
	auto const image_view        = swapchain.GetCurrentImageView();
	auto const new_layout        = vk::ImageLayout::eTransferSrcOptimal;
	auto const image_aspect      = vk::ImageAspectFlagBits::eColor;
	auto const image_subresource = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);

	auto extent            = swapchain.GetExtent();
	auto format_block_size = vk::blockSize(swapchain.GetFormat());

	vk::DeviceSize const image_size = extent.width * extent.height * format_block_size;

	if (image_size >= staging_buffer.GetSize()) {
		staging_buffer.Destroy();
		// clang-format off
		CHECK_VULKAN_RESULT(staging_buffer.Create(device, vma_allocator, {
			.size   = image_size,
			.usage  = vk::BufferUsageFlagBits::eTransferSrc,
			.memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		}));
		// clang-format on
	}

	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));

	cmd.Barrier({
		.image         = image,
		.oldLayout     = vk::ImageLayout::ePresentSrcKHR,
		.newLayout     = new_layout,
		.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands,
		.srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
		.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer,
		.dstAccessMask = vk::AccessFlagBits2::eTransferRead,
	});

	auto region = vk::BufferImageCopy{
		.bufferOffset      = 0,
		.bufferRowLength   = 0,
		.bufferImageHeight = 0,
		.imageSubresource  = image_subresource,
		.imageOffset       = vk::Offset3D{0, 0, 0},
		.imageExtent       = vk::Extent3D{extent.width, extent.height, 1},
	};

	cmd.copyImageToBuffer(image, new_layout, staging_buffer, 1, &region);

	auto cmd_info = vk::CommandBufferSubmitInfo{.commandBuffer = cmd};
	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit2(vk::SubmitInfo2{
		.commandBufferInfoCount = 1,
		.pCommandBufferInfos    = &cmd_info,
	}))
	CHECK_VULKAN_RESULT(queue.waitIdle());

	auto data   = staging_buffer.GetMappedData();
	auto stride = extent.width * format_block_size;
	if (stbi_write_bmp(filename.data(), extent.width, extent.height, format_block_size, data)) {
		// if (stbi_write_png(filename.data(), extent.width, extent.height, format_block_size, data, 0)) {
		std::printf("Saved %s\n", filename.data());
	} else {
		std::printf("Failed to write %s\n", filename.data());
	}
	pending_image_save = false;
}

void SDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		DrawWindow();
		if (pics)
			break;
	} while (true);
}

template <typename Range, typename Proj = std::identity>
constexpr inline auto contains(Range&& range, auto&& value, Proj&& proj = std::identity{}) {
	for (auto&& v : range)
		if (std::invoke(proj, v) == value)
			return true;
	return false;
};

void SDFSample::RunBenchmark(TestOptions const& options) {
	// struct TestData {
	// 	vk::Pipeline pipeline;
	// };
	// WindowManager::WaitEvents();
	// if (window.GetShouldClose()) return;

	// window.SetWindowMode(WindowMode::eFullscreen);
	auto [width, height] = options.resolution;
	window.SetSize(width, height);
	// window.Hide();
	// std::printf("Resizing to %dx%d\n", width, height);
	RecreateSwapchain(width, height);
	// DrawWindow();
	// constexpr u32 kIters = 32;
	constexpr u32 kIters = 16;
	// constexpr u32 kIters = 256;
	// constexpr u32 kMaxTestKinds = u32(SdfFunctionType::eCount);
	// constexpr u32 kMaxTestKinds = 5;
	constexpr u32 kMaxTestKinds = kTestFunctionsCount;

	int first_test{}, last_test = kMaxTestKinds - 1;

	auto is_header = true;
	if (is_header) {
		first_test = *function_id;
		last_test  = first_test;
	}

	SdfFunctionType skip[] = {};

	auto draw = [&](u32 id) {
		// if (is_header) {
		// return DrawWindow(pipelines_header[id]);
		// } else {
		return DrawWindow(pipelines[id]);
		// };
	};

	// constexpr u32 kWarmupCount = 2;
	constexpr u32 kWarmupCount = 0;
	for (u32 t_i = first_test; t_i <= last_test; ++t_i) {
		for (u32 iter = 0; iter < kWarmupCount; ++iter) {
			(void)draw(t_i);
		}
	}

	std::array<std::array<u64, kMaxTestKinds + 10>, kIters> test_times;

	for (u32 t_i = first_test; t_i <= last_test; ++t_i) {
		// TestData& data = test_data[t_i];
		for (u32 iter = 0; iter < kIters; ++iter) {
			// WindowManager::PollEvents();
			u64   time_nanoseconds = draw(t_i);
			float ns_per_tick      = physical_device.GetNsPerTick();
			float elapsed_ms       = (time_nanoseconds * ns_per_tick) / 1e6f;
			test_times[iter][t_i]  = time_nanoseconds;
		}
	}

	std::string_view names[] = {"CoopVec", "WeightsInBuffer", "WeightsInBufferFloat16", "WeightsInHeader"};
	std::string_view fs[]    = {
        "3_16_16_16_1_625",
        "3_24_24_24_1_1321",
        "3_32_32_32_1_2273",
        "3_32_32_32_32_1_3329",
        "3_48_48_48_1_4945",
        "3_64_64_64_1_8641",
        "3_128_128_128_1_33665",
        "3_128_128_128_128_1_50177",
    };

	// Print csv
	// if constexpr (kDstMatrixType == vk::ComponentTypeKHR::eFloat16) {
	// 	std::printf("CoopVec_Float16,ScalarInline_Float16,ScalarBuffer_Float16,Vec4_Float16\n");
	// } else {
	// 	std::printf("ScalarInline_Float32,ScalarBuffer_Float32,Vec4_Float32\n");
	// }

	for (u32 t_i = first_test; t_i <= last_test; ++t_i) {
		if (contains(skip, SdfFunctionType(t_i))) continue;
		if (is_header) {
			// std::printf("SDF_%s_%u", names[u32(function_type)].data(), t_i);
			std::printf("SDF_%s_%s", names[u32(function_type)].data(), fs[t_i].data());
		} else {
			std::printf("%s", names[t_i].data());
		}
		if (t_i < last_test && !contains(skip, SdfFunctionType(t_i + 1))) std::printf(",");
	}
	std::printf("\n");

	for (u32 iter = 0; iter < kIters; ++iter) {
		auto const& tests_row = test_times[iter];

		for (u32 t_i = first_test; t_i <= last_test; ++t_i) {
			if (contains(skip, SdfFunctionType(t_i))) continue;
			std::printf("%llu", tests_row[t_i]);
			// std::printf("%f", float(tests_row[t_i])/1e6f);
			if (t_i < last_test && !contains(skip, SdfFunctionType(t_i + 1))) std::printf(",");
		}
		// std::printf("%llu \n", test_times[iter][last_test]);
		std::printf("\n");
	}
}

auto SDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	auto args_range = std::span(argv + 1, argc - 1);

	if (std::ranges::contains(args_range, std::string_view("--help")))
		return "--help";

	for (auto it = args_range.begin(); it != args_range.end(); ++it) {
		auto arg = std::string_view(*it);
		if (arg == "--benchmark" || arg == "-b") is_test_mode = true;
		else if (arg == "--verbose" || arg == "-v") is_verbose = true;
		else if (arg == "--validation") use_validation = true;
		else if (arg == "--pics" || arg == "-p") pics = true;
		else if (arg == "--kind") {
			if ((it + 1) == args_range.end()) return "expected <kind>";
			auto kind = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(kind.data(), kind.data() + kind.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= std::to_underlying(SdfFunctionType::eCount)) return *(it + 1);
			function_type = static_cast<SdfFunctionType>(value);
			// benchmark_single = true;
			++it;
		} else if (arg == "-f") {
			if ((it + 1) == args_range.end()) return "expected <id>";
			auto str = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(str.data(), str.data() + str.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= kTestFunctionsCount) return *(it + 1);
			// function_type    = SDFFunctionType::eWeightsInHeader;
			// benchmark_single = true;
			function_id = value;
			++it;
		} else return *it;
	}
	return nullptr;
}

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	SDFSample sample;

	if (char const* unknown_arg = sample.ParseArgs(argc, argv); unknown_arg) {
		std::printf("Unknown argument: %s\n", unknown_arg);
		std::printf("Usage: %s [--test] [--verbose] [--validation]\n",
					std::filesystem::path(argv[0]).filename().string().c_str());
		return 0;
	}

	TestOptions options{
		// .resolution = {640, 480},
		.resolution = {1920, 1080},
		// .test_count = 64,
		.test_count = 1,
	};

	int2 res_arr[] = {
		{1920, 1080},
		// {3840, 2160},
		// {512, 512},
		{640, 480},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		{3840, 2160},
	};

	std::string_view weights_files[] = {
		// "Assets/SDFWeights_64_3.bin",

		"Assets/SDFWeights_16_3.bin",
		"Assets/SDFWeights_24_3.bin",
		"Assets/SDFWeights_32_3.bin",
		"Assets/SDFWeights_32_4.bin",
		"Assets/SDFWeights_48_3.bin",
		"Assets/SDFWeights_64_3.bin",
		"Assets/SDFWeights_128_3.bin",
		"Assets/SDFWeights_128_4.bin",
		// "Assets/SDFWeights_128_5.bin",
	};

	auto test_count = //

		// std::size(res_arr);
		// std::size(weights_files);
		// 1;
		4;

	auto is_render_mode = not sample.is_test_mode and not sample.pics;
	if (is_render_mode) {
		test_count = 1;
	}

	// if ()
	for (int i = 0; i < test_count; ++i) {
		if (!is_render_mode) sample.function_id = i;

		sample.Init({weights_files[i]});
		if (sample.IsTestMode()) {
			// options.resolution   = res_arr[i];
			// options.weights_file = weights_files[i];
			// std::printf("resolution: %d x %d\n", res_arr[i].x, res_arr[i].y);
			sample.RunBenchmark(options);

		}

		else if (sample.pics) {
			sample.Run();

			char fname[256] = {};

			std::snprintf(fname, sizeof(fname), "sdf_%d.bmp", *sample.function_id);
			sample.SaveSwapchainImageToFile(fname);
		} else {
			sample.Init({weights_files[sample.function_id.value_or(0)]});
			sample.Run();
		}
		// std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		if (i < test_count - 1)
			sample.Destroy();
	}

	return 0;
}
