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

#include "Shaders/SDFWeights.h"

using namespace Utils;
using namespace mesh;

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

	static constexpr vk::ComponentTypeKHR kSrcComponentType = COMPONENT_TYPE;
	static constexpr vk::ComponentTypeKHR kDstMatrixType    = COMPONENT_TYPE;
	static constexpr vk::ComponentTypeKHR kDstVectorType    = COMPONENT_TYPE;

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
	void CreateSwapchain();

	void RecordCommands(vk::Pipeline pipeline, NetworkOffsets const& offsets);
	// Return time in nanoseconds
	auto GetQueryResult() -> u64;
	// auto DrawWindow(vk::Pipeline pipeline, NetworkOffsets const& offsets) -> u64;
	auto DrawWindow() -> u64;
	// auto DrawWindow() -> u64 { return DrawWindow(pipelines[u32(SdfFunctionType::eScalarBuffer)], row_major_offsets); }
	void RecreateSwapchain(int width, int height);
	void SaveSwapchainImageToFile(std::string_view filename);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }
	bool IsTestMode() const { return bIsTestMode; }

	auto ParseArgs(int argc, char const* argv[]) -> char const*;

	bool bIsTestMode    = false;
	bool bVerbose       = false;
	bool bUseValidation = false;

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

	Camera camera{{
		.position = {-4.180247, -0.427392, 0.877357},
		.focus    = {0.0f, 0.0f, 0.0f},
		.up       = {0.213641, -0.093215, 0.972476},
		.fov      = 35.0f,
		.z_near   = 0.01f,
		.z_far    = 1000.0f,
	}};
};

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
		case Key::eF8:
			sample->SaveSwapchainImageToFile("sdf.bmp");
			// sample->SaveSwapchainImageToFile("sdf.png");
			break;
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

void SDFSample::Init() {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();

	u32 const initial_width = 1600, initial_height = 1200;
	window.Init({.x = 30, .y = 30, .width = initial_width, .height = initial_height, .title = "SDF"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!bIsTestMode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
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
	CreateSwapchain();

	// network_parameters.resize(network.GetParametersSize());

	auto [result, cooperative_vector_properties] = physical_device.getCooperativeVectorPropertiesNV();
	CHECK_VULKAN_RESULT(result);
}

SDFSample::~SDFSample() { Destroy(); }

void SDFSample::Destroy() {

	if (device) {
		CHECK_VULKAN_RESULT(device.waitIdle());

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
		.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
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

	auto [result, index] = physical_device.GetQueueFamilyIndex({.surface = surface});
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
		vk::PhysicalDeviceVulkan11Features{},
		vk::PhysicalDeviceVulkan12Features{},
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

auto SDFSample::DrawWindow() -> u64 {
	return {};
}

void SDFSample::RecordCommands(vk::Pipeline pipeline, NetworkOffsets const& offsets) {
}

void SDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}

void SDFSample::SaveSwapchainImageToFile(std::string_view filename) {
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

void SDFSample::RunTest() {
}

auto SDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	for (std::string_view const arg : std::span(argv + 1, argc - 1)) {
		if (arg == "--test" || arg == "-t") bIsTestMode = true;
		else if (arg == "--verbose") bVerbose = true;
		else if (arg == "--validation") bUseValidation = true;
		else return arg.data();
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

	sample.Init();
	if (sample.IsTestMode()) {
		sample.RunTest();
	} else {
		sample.Run();
	}
	return 0;
}
