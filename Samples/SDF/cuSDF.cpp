
#include <cstdio>
// #include <cuda.h>
#include <cuda_runtime.h>

#include "CheckResult.h"
#include "Shaders/SDFConfig.h"

// #ifdef _WIN64
// #define WIN32_LEAN_AND_MEAN
// #include <windows.h>
// #include <VersionHelpers.h>
// #endif

typedef void* HANDLE;

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
// #include "Shaders/SDFConstants.h"
}

#include "Shaders/SDFWeights.h"

#define _CHECK_VULKAN_RESULT2(func, line) \
	{ \
		::vk::Result local_result_##line = ::vk::Result(func); \
		if (local_result_##line != ::vk::Result::eSuccess) [[unlikely]] { \
			::std::printf("Vulkan error: %s " #func " in " __FILE__ ":" #line, ::vk::to_string(local_result_##line).c_str()); \
			::std::exit(1); \
		} \
	}

#define _CHECK_VULKAN_RESULT(func, line) _CHECK_VULKAN_RESULT2(func, line)
#define CHECK_VULKAN_RESULT(func) _CHECK_VULKAN_RESULT(func, __LINE__)

#define STRINGIFY(x) #x

#define MACRO_FWD(macro, ...) macro(__VA_ARGS__)

#define _CHECK_CUDA_RT(expr, line) \
	{ \
		auto local_result_##line = (expr); \
		if (cudaSuccess != local_result_##line) { \
			char const* error_str_##line = cudaGetErrorString(local_result_##line); \
			::std::fprintf(stderr, "CUDA error: %s " #expr " in " __FILE__ ":" #line, error_str_##line); \
			::std::exit(1); \
		} \
	}

#define CHECK_CUDA_RT(func) MACRO_FWD(_CHECK_CUDA_RT, func, __LINE__)

using namespace Utils;

#ifdef _WIN64
HANDLE GetVkImageMemoryHandle(vk::Device device, vk::DeviceMemory memory, vk::ExternalMemoryHandleTypeFlagBitsKHR type) {
	vk::MemoryGetWin32HandleInfoKHR info = {.memory = memory, .handleType = type};

	HANDLE handle;
#if 0
	CHECK_VULKAN_RESULT(device.getMemoryWin32HandleKHR(&info, &handle));
#endif
	return handle;
}

HANDLE GetVkSemaphoreHandle(vk::Device device, vk::Semaphore semaphore, vk::ExternalSemaphoreHandleTypeFlagBitsKHR type) {
	vk::SemaphoreGetWin32HandleInfoKHR info = {.semaphore = semaphore, .handleType = type};

	HANDLE handle;
#if 0
	CHECK_VULKAN_RESULT(device.getSemaphoreWin32HandleKHR(&info, &handle));
#endif
	// auto [result, handle] = device.getSemaphoreWin32HandleKHR({.semaphore = semaphore, .handleType = type});
	// CHECK_VULKAN_RESULT(result);
	return handle;
}
#else

#endif

auto ImportVkSemaphore(vk::Device device, vk::Semaphore sem) -> cudaExternalSemaphore_t {
	cudaExternalSemaphoreHandleDesc desc{
#ifdef _WIN64
		.type   = cudaExternalSemaphoreHandleTypeOpaqueWin32,
		.handle = {.win32 = {.handle = GetVkSemaphoreHandle(device, sem, vk::ExternalSemaphoreHandleTypeFlagBitsKHR::eOpaqueWin32)}},
#else
		.type   = cudaExternalSemaphoreHandleTypeOpaqueFd,
		.handle = {.fd = getVkSemaphoreHandle(device, sem, vk::ExternalSemaphoreHandleTypeFlagBitsKHR::eOpaqueFd)},
#endif
	};
	cudaExternalSemaphore_t ret;
	CHECK_CUDA_RT(cudaImportExternalSemaphore(&ret, &desc));
	return ret;
}

struct PhysDevice : public VulkanRHI::PhysicalDevice {
	PhysDevice() {
	}
	bool IsSuitable(vk::SurfaceKHR const& surface, std::span<char const* const> extensions) {
		bool const supports_extensions =
			SupportsExtensions(extensions);
		bool const supports_queues =
			SupportsQueue({.flags = vk::QueueFlagBits::eGraphics, .surface = surface});
		if (supports_extensions && supports_queues) {
			return true;
		}
		return false;
	}

	inline float GetNsPerTick() const { return static_cast<float>(GetProperties10().limits.timestampPeriod); }
};
class CuSDFSample;

struct CudaContext {
	// Cuda
	cudaExternalMemory_t external_memory = nullptr;
	cudaMipmappedArray_t mipmap_array    = nullptr;
	cudaArray_t          array           = nullptr;
	cudaSurfaceObject_t  surface         = 0;
	cudaStream_t         stream          = nullptr;

	std::vector<cudaExternalSemaphore_t> wait_semaphores;
	std::vector<cudaExternalSemaphore_t> signal_semaphores;

	void Init(CuSDFSample const& sample);

	void ImportImages(CuSDFSample const& sample);

	void Render(vk::CommandBuffer cmd);
};

class CuSDFSample {
public:
	static constexpr u32 kApiVersion = vk::ApiVersion13;

	static constexpr char const* kEnabledLayers[] = {
		"VK_LAYER_KHRONOS_validation",
	};
	static constexpr char const* kEnabledDeviceExtensions[] = {
#ifdef _WIN64
		vk::KHRExternalMemoryWin32ExtensionName,
		vk::KHRExternalSemaphoreWin32ExtensionName,
#else
		vk::KHRExternalMemoryFdExtensionName,
		vk::KHRExternalSemaphoreFdExtensionName,
#endif
		vk::KHRSwapchainExtensionName,
	};

	static constexpr vk::ComponentTypeKHR kSrcComponentType = COMPONENT_TYPE;
	static constexpr vk::ComponentTypeKHR kDstMatrixType    = COMPONENT_TYPE;
	static constexpr vk::ComponentTypeKHR kDstVectorType    = COMPONENT_TYPE;

	static constexpr u32 kFramesInFlight = 3;

	static constexpr u32 kNetworkLayers = 4;

	// struct NetworkOffsets {
	// 	u32 weights_offsets[kNetworkLayers];
	// 	u32 biases_offsets[kNetworkLayers];
	// };

	~CuSDFSample();

	void Init();
	// void InitCuda();
	// void DestroyCuda();
	void Run();
	void RunTest();
	void Destroy();
	void CreateInstance();
	void SelectPhysicalDevice();
	void GetPhysicalDeviceInfo();
	void CreateDevice();

	void RecordCommands();
	auto DrawWindow() -> u64;
	void RecreateSwapchain(int width, int height);
	void SaveSwapchainImageToFile(std::string_view filename);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }
	bool IsTestMode() const { return is_test_mode; }

	auto ParseArgs(int argc, char const* argv[]) -> char const*;

	bool is_test_mode   = false;
	bool is_verbose     = false;
	bool use_validation = false;

	GLFWWindow window{};
	Mouse      mouse;

	vk::Instance                    instance{};
	vk::AllocationCallbacks const*  allocator{nullptr};
	VmaAllocator                    vma_allocator{};
	vk::PipelineCache               pipeline_cache{nullptr};
	vk::DebugUtilsMessengerEXT      debug_messenger{};
	std::span<char const* const>    enabled_layers{};
	std::vector<vk::PhysicalDevice> vulkan_physical_devices{};
	PhysDevice                      physical_device{};
	vk::Device                      device{};
	VulkanRHI::Swapchain            swapchain{};
	bool                            swapchain_dirty = false;
	vk::SurfaceKHR                  surface{};

	vk::Queue queue{};
	u32       queue_family_index = ~0u;

	bool timestamps_supported = false;

	CudaContext cuda;

	Camera camera{{
		.position = {-4.180247, -0.427392, 0.877357},
		.focus    = {0.0f, 0.0f, 0.0f},
		.up       = {0.213641, -0.093215, 0.972476},
		.fov      = 35.0f,
		.z_near   = 0.01f,
		.z_far    = 1000.0f,
	}};
};

namespace {
void FramebufferSizeCallback(GLFWWindow* window, int width, int height) {
	CuSDFSample* sample = static_cast<CuSDFSample*>(window->GetUserPointer());

	sample->swapchain_dirty = true;
	if (width <= 0 || height <= 0) return;
	sample->RecreateSwapchain(width, height);
	sample->camera.updateProjection(width, height);
}

void WindowRefreshCallback(GLFWWindow* window) {
	CuSDFSample* sample = static_cast<CuSDFSample*>(window->GetUserPointer());

	int x, y, width, height;
	window->GetRect(x, y, width, height);
	if (width <= 0 || height <= 0) return;
	sample->DrawWindow();
}

void CursorPosCallback(GLFWWindow* window, double xpos, double ypos) {
	CuSDFSample* sample = static_cast<CuSDFSample*>(window->GetUserPointer());

	sample->mouse.delta_x = static_cast<float>(xpos - sample->mouse.x);
	sample->mouse.delta_y = -static_cast<float>(ypos - sample->mouse.y);
	sample->mouse.x       = static_cast<float>(xpos);
	sample->mouse.y       = static_cast<float>(ypos);

	ProcessViewportInput(sample->window, sample->camera, sample->mouse, sample->mouse.delta_x, sample->mouse.delta_y);
}

void KeyCallback(GLFWWindow* window, int key, int scancode, int action, int mods) {
	CuSDFSample* sample = static_cast<CuSDFSample*>(window->GetUserPointer());
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

void MouseButtonCallback(GLFWWindow* in_window, int in_button, int in_action, int in_mods) {
	CuSDFSample* sample = static_cast<CuSDFSample*>(in_window->GetUserPointer());
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

void CuSDFSample::Init() {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();

	u32 const initial_width = 1600, initial_height = 1200;
	window.Init({.x = 30, .y = 30, .width = initial_width, .height = initial_height, .title = "SDF"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!is_test_mode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}

	int x, y, width, height;
	window.GetFullScreenRect(x, y, width, height);

	camera.updateProjection(initial_width, initial_height);

	auto& input = window.GetInputCallbacks();

	input.cursorPosCallback   = CursorPosCallback;
	input.keyCallback         = KeyCallback;
	input.mouseButtonCallback = MouseButtonCallback;
	window.SetUserPointer(this);

	CreateInstance();

	CHECK_VULKAN_RESULT(WindowManager::CreateWindowSurface(
		instance, static_cast<GLFWwindow*>(window.GetHandle()),
		GetAllocator(), &surface));

	SelectPhysicalDevice();
	GetPhysicalDeviceInfo();

	CreateDevice();

	// Create Swapchain
	window.GetRect(x, y, width, height);
	VulkanRHI::SwapchainInfo info{
		.surface            = surface,
		.extent             = {.width = static_cast<u32>(width), .height = static_cast<u32>(height)},
		.queue_family_index = queue_family_index,
		.frames_in_flight   = kFramesInFlight,
	};
	CHECK_VULKAN_RESULT(swapchain.Create(device, physical_device, info, GetAllocator()));
}

struct CudaImage {
	cudaExternalMemory_t external_memory = nullptr;
	cudaMipmappedArray_t mipmap_array    = nullptr;
	cudaArray_t          array           = nullptr;
	cudaSurfaceObject_t  surface         = 0;
};

void CudaContext::Init(CuSDFSample const& sample) {
	CHECK_CUDA_RT(cudaSetDevice(0));
	CHECK_CUDA_RT(cudaStreamCreate(&stream));

	// int x, y, width, height;
	// sample.window.GetRect(x, y, width, height);

	// Import semaphores
	auto swapchain_images = sample.swapchain.GetImages();
	auto frames           = sample.swapchain.GetFrameData();

	wait_semaphores.resize(swapchain_images.size());
	signal_semaphores.resize(swapchain_images.size());
	for (size_t i = 0; i < swapchain_images.size(); ++i) {
		wait_semaphores[i]   = ImportVkSemaphore(sample.device, frames[i].GetImageAvailableSemaphore());
		signal_semaphores[i] = ImportVkSemaphore(sample.device, frames[i].GetRenderFinishedSemaphore());
	}

	// Import images
	auto images = sample.swapchain.GetImages();

	auto extent = sample.swapchain.GetExtent();

	// Create cudaExternalMemory_t

	for (size_t i = 0; i < images.size(); ++i) {
		CudaImage                            cuda_image;
		cudaExternalMemoryMipmappedArrayDesc desc{
			.offset     = 0,
			.formatDesc = cudaChannelFormatDesc{.x = 8, .y = 8, .z = 8, .w = 8, .f = cudaChannelFormatKind::cudaChannelFormatKindUnsigned},
			.extent     = {.width = extent.width, .height = extent.height, .depth = 1},
			.numLevels  = 1,
		};
		CHECK_CUDA_RT(cudaExternalMemoryGetMappedMipmappedArray(&cuda_image.mipmap_array, cuda_image.external_memory, &desc));
		CHECK_CUDA_RT(cudaGetMipmappedArrayLevel(&cuda_image.array, cuda_image.mipmap_array, 0));
		cudaResourceDesc resource_desc{
			.resType = cudaResourceType::cudaResourceTypeArray,
			.res     = {.array = {.array = cuda_image.array}},
		};
		CHECK_CUDA_RT(cudaCreateSurfaceObject(&cuda_image.surface, &resource_desc));
	}
};

CuSDFSample::~CuSDFSample() { Destroy(); }

void CuSDFSample::Destroy() {

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

// void CuSDFSample::InitCuda() {
// };

// void CuSDFSample::DestroyCuda() {};

void CuSDFSample::CreateInstance() {
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

void CuSDFSample::SelectPhysicalDevice() {
	for (vk::PhysicalDevice const& device : vulkan_physical_devices) {
		physical_device.Assign(device);
		CHECK_VULKAN_RESULT(physical_device.GetDetails());
		if (physical_device.IsSuitable(surface, kEnabledDeviceExtensions)) {
			if (is_verbose) {
			}
			return;
		}
	}

	std::printf("No suitable physical device found\n");
	// std::this_thread::sleep_for(std::chrono::milliseconds(2000));
	std::getchar();
	std::exit(1);
}

void CuSDFSample::GetPhysicalDeviceInfo() {
}

void CuSDFSample::CreateDevice() {
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

// Draw to swapchain image
auto CuSDFSample::DrawWindow() -> u64 {
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
	RecordCommands();
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;
	return {};
}

void CuSDFSample::RecordCommands() {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	vk::Rect2D               render_rect{{0, 0}, {static_cast<u32>(width), static_cast<u32>(height)}};
	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	// cmd.resetQueryPool(timestamp_query_pool, GetCurrentTimestampIndex(), kTimestampsPerFrame);
	// cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timestamp_query_pool, GetCurrentTimestampIndex());

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

	camera.updateProjectionViewInverse();

	// Render

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
	CHECK_VULKAN_RESULT(cmd.end());
}
/*
void CudaContext::Render(vk::CommandBuffer cmd, cudaExternalSemaphore_t wait, cudaExternalSemaphore_t signal) {
	// Wait for Vulkan semaphore in CUDA
	cudaExternalSemaphoreWaitParams wait_params = {};
	CHECK_CUDA_RT(cudaWaitExternalSemaphoresAsync(&wait, &wait_params, 1, stream))

	// Launch CUDA kernel
	dim3 block_size(16, 16);
	dim3 grid_size((width + block_size.x - 1) / block_size.x,
				   (height + block_size.y - 1) / block_size.y);

	// rayMarchKernel<<<grid_size, block_size, 0, stream>>>(
	// 	surface, width, height,
	// 	camera_pos, camera_forward, camera_right, camera_up, fov);
	CHECK_CUDA_RT(cudaGetLastError());

	// Signal semaphore when CUDA is done
	cudaExternalSemaphoreSignalParams signal_params = {};
	CHECK_CUDA_RT(cudaSignalExternalSemaphoresAsync(&signal, &signal_params, 1, stream))
}
 */
void CuSDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}

void CuSDFSample::SaveSwapchainImageToFile(std::string_view filename) {
}

void CuSDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		DrawWindow();
	} while (true);
}

void CuSDFSample::RunTest() {
}

auto CuSDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	for (std::string_view const arg : std::span(argv + 1, argc - 1)) {
		if (arg == "--test" || arg == "-t") is_test_mode = true;
		else if (arg == "--verbose") is_verbose = true;
		else if (arg == "--validation") use_validation = true;
		else return arg.data();
	}
	return nullptr;
}

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	CuSDFSample sample;

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
