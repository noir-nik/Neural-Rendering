#include <cstddef>

#include "CheckResult.h"
#include "Shaders/BRDFConfig.h"

import NeuralGraphics;
import vulkan_hpp;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;
import Math;
import std;

using namespace math;
using namespace mesh;
extern "C++" {
#include "Shaders/BRDFConstants.h"
}

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t

using namespace Utils;

using VulkanRHI::Buffer;
using VulkanRHI::Image;

class BRDFSample {
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

	// vk::ComponentTypeKHR kSrcComponentType = vk::ComponentTypeKHR::eFloat32;
	// vk::ComponentTypeKHR kDstMatrixType    = vk::ComponentTypeKHR::eFloat32;
	// vk::ComponentTypeKHR kDstVectorType    = vk::ComponentTypeKHR::eFloat32;

	static constexpr u32 kFramesInFlight = 3;

	using Vertex = mesh::UVSphere::Vertex;

	~BRDFSample();

	void Init();
	void Run();
	void RunTest() {};
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
		BrdfFunctionType function_type = BrdfFunctionType::eCoopVec) -> vk::Pipeline;

	// void BuildNetwork();
	void CreateAndUploadBuffers();

	// Return time in nanoseconds
	auto GetQueryResult() -> u64;
	void RecreateSwapchain(int width, int height);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }

	bool IsTestMode() const { return bIsTestMode; }

	auto ParseArgs(int argc, char const* argv[]) -> char const*;

	bool bIsTestMode    = false;
	bool verbose        = false;
	bool bUseValidation = true;

	GLFWWindow window{};
	struct {
		float x = 300.0f;
		float y = 300.0f;

		float        delta_x                                                                               = 0.0f;
		float        delta_y                                                                               = 0.0f;
		Glfw::Action button_state[std::underlying_type_t<Glfw::MouseButton>(Glfw::MouseButton::eLast) + 1] = {};

		bool is_dragging = false;
		void StartDragging() { is_dragging = true; }
		void StopDragging() { is_dragging = false; }
	} mouse;
	void ProcessViewportInput();

	vk::Instance                    instance;
	vk::AllocationCallbacks const*  allocator     = nullptr;
	VmaAllocator                    vma_allocator = nullptr;
	vk::PipelineCache               pipeline_cache;
	vk::DebugUtilsMessengerEXT      debug_messenger;
	std::span<char const* const>    enabled_layers;
	std::vector<vk::PhysicalDevice> vulkan_physical_devices;
	PhysicalDevice                  physical_device;
	vk::Device                      device;
	VulkanRHI::Swapchain            swapchain;
	bool                            swapchain_dirty = false;
	vk::SurfaceKHR                  surface;

	vk::Queue queue;
	u32       queue_family_index = ~0u;

	bool timestamps_supported = false;

	vk::DescriptorSetLayout descriptor_set_layout;
	vk::DescriptorPool      descriptor_pool;
	vk::DescriptorSet       descriptor_set;

	vk::PipelineLayout pipeline_layout;

	// vk::Pipeline coopvec_pipeline;
	// vk::Pipeline scalar_inline_pipeline;
	// vk::Pipeline scalar_buffer_pipeline;
	// vk::Pipeline vec4_pipeline;

	std::array<vk::Pipeline, u32(BrdfFunctionType::eCount)> pipelines;

	Buffer         device_buffer;
	vk::DeviceSize vertices_offset = 0;
	vk::DeviceSize indices_offset  = 0;
	vk::DeviceSize weights_offset  = 0;
	// Buffer x_buffer;
	Buffer staging_buffer;
	// Buffer brdf_weights_buffer;

	// float    radius   = 1.0f;
	// u32      segments = 8;
	// u32      rings    = 8;
	UVSphere sphere = UVSphere(1.0f, 32, 16);

	Image depth_image;
	bool  use_depth = true;

	VulkanCoopVecNetwork optimal_network = {
		Linear(6, 64),
		Linear(64, 64),
		Linear(64, 64),
		Linear(64, 6),
		Linear(6, 3),
	};

	VulkanCoopVecNetwork linear_network = {
		Linear(6, 64),
		Linear(64, 64),
		Linear(64, 64),
		Linear(64, 6),
		Linear(6, 3),
	};

	static constexpr u32 kNetworkLinearLayers = 5;

	// struct NetworkOffsets {
	// 	u32 weights_offsets[kNetworkLinearLayers];
	// 	u32 biases_offsets[kNetworkLinearLayers];
	// };

	// NetworkOffsets optimal_offsets;
	// NetworkOffsets row_major_offsets;

	// NetworkOffsets* default_offsets = nullptr;

	void RecordCommands(vk::Pipeline pipeline, VulkanCoopVecNetwork const& network);
	auto DrawWindow(vk::Pipeline pipeline, VulkanCoopVecNetwork const& network) -> u64;

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

	Camera camera{{
		.position = {1.0f, 3.0f, 5.0f},
		.fov      = 50.0f,
		.z_near   = 0.01f,
		.z_far    = 1000.0f,
	}};
};

class WeightsLoader {
public:
	constexpr static char const kHeader[] = "hydrann1";
	WeightsLoader(std::string_view filename) { Init(filename); }

	uint32_t NextRows() const { return rows; } // returns 0 on error
	uint32_t NextCols() const { return cols; } // returns 0 on error
	bool     LoadNext(float* weights, float* bias);
	bool     HasNext() const { return layers > 0; }

	auto GetFileSize() const -> std::size_t { return file_size; }

private:
	std::ifstream file;
	uint32_t      rows      = 0;
	uint32_t      cols      = 0;
	uint32_t      layers    = 0;
	std::size_t   file_size = 0;

	// Return file size
	std::size_t Init(std::string_view filename);

	void ReadNextLayerInfo();
};

std::size_t WeightsLoader::Init(std::string_view filename) {
	file.open(filename.data(), std::ios::ate | std::ios::binary);
	file_size = static_cast<std::size_t>(file.tellg());

	// Check header
	auto header_size = std::size(kHeader);
	if (file_size < header_size) return 0;
	file.seekg(0);
	char buf[header_size];
	file.read(buf, header_size - 1);
	buf[header_size - 1] = '\0';
	if (std::string_view(kHeader) != buf) return 0;

	file.read(reinterpret_cast<char*>(&layers), sizeof(uint32_t));
	if (!file.good()) {
		layers = 0;

		file.close();

		return 0;
	} else {
		ReadNextLayerInfo();
		return 1;
	}
}

bool WeightsLoader::LoadNext(float* weights, float* bias) {
	if (layers == 0) return false;

	const uint32_t count = rows * cols;
	if (count > 0) {
		file.read(reinterpret_cast<char*>(weights), count * sizeof(float));
		file.read(reinterpret_cast<char*>(bias), rows * sizeof(float));
	}

	if (!file) {
		layers = 0;
		rows = cols = 0;
		return false;
	}

	layers -= 1;
	ReadNextLayerInfo();
	return true;
}

void WeightsLoader::ReadNextLayerInfo() {
	if (layers == 0) {
		rows = cols = 0;
		return;
	}

	file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
	file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
	if (!file.good()) {
		layers = 0;
	}
}

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
	sample->DrawWindow(sample->pipelines[u32(BrdfFunctionType::eDefault)], sample->linear_network);
}

void PrintMat4(float4x4 const& mat) {
	for (int i = 0; i < 4; ++i) {
		std::printf("%f %f %f %f\n", mat[i][0], mat[i][1], mat[i][2], mat[i][3]);
	}
}

void BRDFSample::ProcessViewportInput() {
	using namespace Glfw;

	auto&       camera_pos     = camera.getPosition();
	auto const& camera_right   = camera.getRight();
	auto const& camera_up      = camera.getUp();
	auto const& camera_forward = camera.getForward();

	vec2 delta_pos = {-mouse.delta_x, mouse.delta_y};
	// vec2 delta_pos = {mouse.delta_x, mouse.delta_y};

	GLFWwindow* glfw_window = static_cast<GLFWwindow*>(window.GetHandle());

	u32 const pressed_buttons_count =
		std::ranges::fold_left(
			std::span(mouse.button_state) | std::views::take(3 /* left, middle, right */),
			0, [](u32 state, Action action) {
				return state + (action == Action::ePress);
			});

	if (pressed_buttons_count > 1) [[unlikely]]
		return;
	auto button_pressed = [this](MouseButton button) { return mouse.button_state[std::to_underlying(button)] == Action::ePress; };

	if (button_pressed(MouseButton::eRight)) {
		// if (glfwGetKey(glfw_window, std::to_underlying(Glfw::Key::eLeftAlt)) == std::to_underlying(Glfw::Action::ePress)) {
		if (mouse.is_dragging) {
			auto zoom_factor = camera.zoom_factor * length(camera_pos - camera.focus);
			auto movement    = (zoom_factor * delta_pos.x) * camera_forward;
			camera.view      = camera.view | translate(movement);
			// camera.focus += movement;
		} else {
			camera_pos -= camera.focus;
			// camera.view = rotate4x4(camera_up, -delta_pos.x * camera.rotation_factor)
			// 	 * rotate4x4(camera_right, delta_pos.y * camera.rotation_factor)
			// 	 * camera.view; // trackball

			// Correct upside down
			float rotation_sign = std::copysignf(1.0f, camera.getUp().y);

			camera.view = camera.view
						  | rotate(camera_right, delta_pos.y * camera.rotation_factor)
						  | rotateY(rotation_sign * delta_pos.x * camera.rotation_factor);
			camera_pos += camera.focus;
		}
		// window->AddFramesToDraw(1);
		// camera.updateProjView();
		return;
	}

	if (button_pressed(MouseButton::eMiddle)) {

		int x, y, width, height;
		window.GetRect(x, y, width, height);
		camera.moveFromMouse(width, height, delta_pos.x, delta_pos.y);
	}

	if (button_pressed(MouseButton::eLeft)) {
		// window->AddFramesToDraw(1);
		return;
	}
}
static void CursorPosCallback(GLFWWindow* window, double xpos, double ypos) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	sample->mouse.delta_x = static_cast<float>(xpos - sample->mouse.x);
	sample->mouse.delta_y = static_cast<float>(ypos - sample->mouse.y);
	sample->mouse.x       = static_cast<float>(xpos);
	sample->mouse.y       = static_cast<float>(ypos);

	sample->ProcessViewportInput();

	// std::printf("mouse position: (%f, %f), delta: (%f, %f)\n", sample->mouse.x, sample->mouse.y,
	// 			sample->mouse.delta_x, sample->mouse.delta_y);
}

static void KeyCallback(GLFWWindow* window, int key, int scancode, int action, int mods) {
	BRDFSample* sample = static_cast<BRDFSample*>(window->GetUserPointer());

	switch (key) {
	case 256:
		window->SetShouldClose(true);
		break;
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

void BRDFSample::Init() {
	WindowManager::SetErrorCallback(WindowErrorCallback);
	WindowManager::Init();
	window.Init({.x = 30, .y = 30, .width = 800, .height = 600, .title = "BRDF Sample"});
	window.GetWindowCallbacks().framebufferSizeCallback = FramebufferSizeCallback;
	if (!bIsTestMode) {
		window.GetWindowCallbacks().windowRefreshCallback = WindowRefreshCallback;
	}

	int x, y, width, height;
	window.GetFullScreenRect(x, y, width, height);

	camera.updateProjection(800, 600);

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
	CreateAndUploadBuffers();

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

void BRDFSample::CreateInstance() {
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
		.enabledLayerCount       = static_cast<u32>(std::size(enabled_layers)),
		.ppEnabledLayerNames     = enabled_layers.data(),
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

	std::optional<std::vector<std::byte>> shader_codes[] = {
		ReadBinaryFile("Shaders/BRDFMain.slang.spv").or_else([] {
			std::printf("Failed to read shader file!\n");
			std::exit(1);
			return std::optional<std::vector<std::byte>>{};
		}),
	};

	vk::ShaderModuleCreateInfo shader_module_infos[std::size(shader_codes)];
	vk::ShaderModule           shader_modules[std::size(shader_codes)];

	for (u32 i = 0; i < std::size(shader_codes); ++i) {
		shader_module_infos[i].codeSize = (*shader_codes[i]).size();
		shader_module_infos[i].pCode    = reinterpret_cast<const u32*>((*shader_codes[i]).data());
		CHECK_VULKAN_RESULT(device.createShaderModule(&shader_module_infos[i], GetAllocator(), &shader_modules[i]));
	}

	pipelines[int(BrdfFunctionType::eDefault)] = CreatePipeline(shader_modules[0], BrdfFunctionType::eDefault);
	pipelines[int(BrdfFunctionType::eCoopVec)] = CreatePipeline(shader_modules[0], BrdfFunctionType::eCoopVec);

	for (auto& shader_module : shader_modules) {
		device.destroyShaderModule(shader_module, GetAllocator());
	}
}

auto BRDFSample::CreatePipeline(vk::ShaderModule shader_module, BrdfFunctionType function_type) -> vk::Pipeline {

	// Specialization constant for type of inferencing function
	BrdfFunctionType specialization_value = function_type;

	vk::SpecializationMapEntry specialization_entry{
		.constantID = 0,
		.offset     = 0,
		.size       = sizeof(BrdfFunctionType),
	};
	vk::SpecializationInfo specialization_info{
		.mapEntryCount = 1,
		.pMapEntries   = &specialization_entry,
		.dataSize      = sizeof(BrdfFunctionType),
		.pData         = &specialization_value,
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
		.depthAttachmentFormat   = depth_image.GetFormat(),
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
		.pDepthStencilState  = &depth_stencil_state,
		.pColorBlendState    = &color_blend_state,
		.pDynamicState       = &dynamic_state,
		.layout              = pipeline_layout,
	};

	vk::Pipeline pipeline;
	CHECK_VULKAN_RESULT(device.createGraphicsPipelines(GetPipelineCache(), 1, &info, GetAllocator(), &pipeline));

	return pipeline;
}

void DumpVertexData(std::span<const UVSphere::Vertex> vertices, std::span<const UVSphere::IndexType> indices) {
	std::printf("Vertices:\n");
	for (u32 i = 0; i < vertices.size(); ++i) {
		const auto& vertex = vertices[i];
		std::printf("%u) pos = (%f, %f, %f), uv = (%f, %f), normal = (%f, %f, %f)\n", i, vertex.pos[0], vertex.pos[1], vertex.pos[2], vertex.u, vertex.v, vertex.normal[0], vertex.normal[1], vertex.normal[2]);
	}
	std::printf("Indices:\n");
	for (u32 i = 0; i < indices.size() / 3; ++i) {
		std::printf("%u) %u, %u, %u\n", i, indices[i * 3], indices[i * 3 + 1], indices[i * 3 + 2]);
	}
}

void BRDFSample::CreateAndUploadBuffers() {
	std::size_t vertices_size_bytes   = sphere.GetVertexCount() * sizeof(UVSphere::Vertex);
	std::size_t indices_size_bytes    = sphere.GetIndexCount() * sizeof(UVSphere::IndexType);
	std::size_t alignment             = sizeof(float) * 4;
	std::size_t vertices_size_aligned = AlignUpPowerOfTwo(vertices_size_bytes, alignment);
	std::size_t indices_size_aligned  = AlignUpPowerOfTwo(indices_size_bytes, alignment);

	// auto brdf_weights_opt =
	// 	ReadBinaryFile(weights_path)
	// 		.or_else([] -> std::invoke_result_t<decltype(ReadBinaryFile), decltype("")> {
	// 			std::printf("Failed to read BRDF weights file!\n");
	// 			std::exit(1);
	// 			return std::nullopt;
	// 		});

	// auto brdf_weights = std::span<std::byte>(*brdf_weights_opt);

	auto weights_path = "Assets/simple_brdf_weights.bin";

	WeightsLoader loader{weights_path};

	std::vector<float> brdf_weights_vec(loader.GetFileSize() / sizeof(float));

	struct Layer {
		int inputs;
		int outputs;
	};
	Layer layers;

	float* src_weights = brdf_weights_vec.data();
	while (loader.HasNext()) {

		std::size_t const rows = loader.NextRows();
		std::size_t const cols = loader.NextCols();

		std::size_t const weights_count = rows * cols;
		std::size_t const biases_count  = rows;
		// std::printf("Rows, cols: %u %u\n", rows, cols);

		bool loaded = loader.LoadNext(src_weights, src_weights + weights_count);
		if (!loaded) {
			std::printf("Error loading weights\n");
			std::exit(1);
		}
		src_weights += weights_count + biases_count;
	}
	// src_weights = brdf_weights.data();

	using Component = vk::ComponentTypeKHR;
	using Layout    = vk::CooperativeVectorMatrixLayoutNV;

	CHECK_VULKAN_RESULT(linear_network.UpdateOffsetsAndSize(device, Layout::eRowMajor, Component::eFloat32, Component::eFloat32));
	CHECK_VULKAN_RESULT(optimal_network.UpdateOffsetsAndSize(device, Layout::eInferencingOptimal, Component::eFloat16, Component::eFloat16));

	// default_offsets = &row_major_offsets;

	vk::DeviceSize const kMinSize = 256 * 1024;

	std::size_t total_size_bytes = std::max(
		kMinSize,
		optimal_network.GetParametersSize() + linear_network.GetParametersSize()
			+ vertices_size_aligned + indices_size_aligned);

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
		.usage  = vk::BufferUsageFlagBits::eTransferSrc,
		.memory = vk::MemoryPropertyFlagBits::eHostVisible 
				| vk::MemoryPropertyFlagBits::eHostCoherent,
	}));
	// clang-format on

	// Update descriptor set
	vk::DescriptorBufferInfo buffer_infos[] = {
		{.buffer = device_buffer, .offset = 0, .range = device_buffer.GetSize()},
	};

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

	// std::memcpy(staging_buffer.GetMappedData(), vertices.data(), vertices_size_bytes);
	this->vertices_offset = 0;
	this->indices_offset  = this->vertices_offset + vertices_size_aligned;
	this->weights_offset  = this->indices_offset + indices_size_aligned;

	auto p_staging  = static_cast<std::byte*>(staging_buffer.GetMappedData());
	auto p_vertices = reinterpret_cast<UVSphere::Vertex*>(p_staging + this->vertices_offset);
	auto p_indices  = reinterpret_cast<UVSphere::IndexType*>(p_staging + this->indices_offset);
	sphere.WriteVertices(p_vertices);
	sphere.WriteIndices(p_indices);

	linear_network.Print();

	std::printf("Total parameters size = %zu\n", linear_network.GetParametersSize());
	// std::size_t offset = weights_offset;
	// update_and_write(offset, vk::CooperativeVectorMatrixLayoutNV::eInferencingOptimal);
	// offset += optimal_size_bytes;
	// update_and_write(offset, vk::CooperativeVectorMatrixLayoutNV::eRowMajor);

	std::printf("Weights offsets, %llu\n", this->weights_offset);
	std::printf("PR %llu\n", 36768 + this->weights_offset - (3 * sizeof(float)));

	// std::printf("Biases offsets, last layer = %u\n", this->row_major_offsets.biases_offsets[kNetworkLinearLayers - 1]);

	auto write_weights = [&](void const*                         src,
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

	// auto const* src = brdf_weights.data();
	std::byte* dst_weights = p_staging + this->weights_offset;

	// auto layer = linear_network.GetLayer<Linear>(0);
	// write_weights(src, layer.GetWeightsSize(), dst, Layout::eRowMajor, Component::eFloat32, Component::eFloat32, layer);

	auto brdf_weights = std::span{reinterpret_cast<std::byte*>(brdf_weights_vec.data()), brdf_weights_vec.size() * sizeof(float)};

	auto src_component_size = sizeof(float);

	auto src_offset = std::size_t{0};
	for (u32 i = 0; i < kNetworkLinearLayers; ++i) {
		auto& linear_layer = linear_network.GetLayer<Linear>(i);

		std::size_t const src_weights_size_bytes = linear_layer.GetWeightsCount() * src_component_size;
		std::size_t const src_biases_size_bytes  = linear_layer.GetBiasesCount() * src_component_size;

		auto const* src_weights = brdf_weights.data() + src_offset;
		// std::byte*  dst_weights = dst_weights_base;
		write_weights(src_weights, src_weights_size_bytes, dst_weights, Layout::eRowMajor, Component::eFloat32, Component::eFloat32, linear_layer);
		src_offset += src_weights_size_bytes;

		auto const* src_bias = brdf_weights.data() + src_offset;
		std::byte*  dst_bias = dst_weights + linear_layer.GetBiasesOffset();
		std::memcpy(dst_bias, src_bias, src_biases_size_bytes);
		src_offset += src_biases_size_bytes;
	}

	linear_network.PrintLayerBiases(kNetworkLinearLayers - 1, Component::eFloat32, p_staging + this->weights_offset);

	// update_and_write(offset, vk::CooperativeVectorMatrixLayoutNV::eInferencingOptimal);

	// DumpVertexData({p_vertices, sphere.GetVertexCount()}, {p_indices, sphere.GetIndexCount()});

	vk::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	vk::BufferCopy regions[] = {
		{
			.srcOffset = vertices_offset,
			.dstOffset = vertices_offset,
			.size      = vertices_size_bytes,
		},
		{
			.srcOffset = indices_offset,
			.dstOffset = indices_offset,
			.size      = indices_size_bytes,
		},
		// Linear
		{
			.srcOffset = weights_offset,
			.dstOffset = weights_offset,
			.size      = linear_network.GetParametersSize(),
		},
	};

	cmd.copyBuffer(staging_buffer, device_buffer, std::size(regions), regions);

	CHECK_VULKAN_RESULT(cmd.end());
	CHECK_VULKAN_RESULT(queue.submit({{.commandBufferCount = 1, .pCommandBuffers = &cmd}}));
	CHECK_VULKAN_RESULT(queue.waitIdle());
}

auto BRDFSample::GetQueryResult() -> u64 {
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

auto BRDFSample::DrawWindow(vk::Pipeline pipeline, VulkanCoopVecNetwork const& network) -> u64 {
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
	device.resetCommandPool(swapchain.GetCurrentCommandPool());
	if (!HandleSwapchainResult(swapchain.AcquireNextImage())) return 0ull;
	CHECK_VULKAN_RESULT(device.resetFences(1, &swapchain.GetCurrentFence()));
	RecordCommands(pipeline, network);
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;

	u64 elapsed = GetQueryResult();
	swapchain.EndFrame();
	return elapsed;
}

void BRDFSample::RecordCommands(vk::Pipeline pipeline, VulkanCoopVecNetwork const& network) {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	auto depth_extent = depth_image.GetExtent();
	if (width > depth_extent.width || height > depth_extent.height) {
		depth_image.Recreate({static_cast<u32>(width), static_cast<u32>(height), 1});
	}

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
			.loadOp      = vk::AttachmentLoadOp::eClear,
			.storeOp     = vk::AttachmentStoreOp::eStore,
			.clearValue  = {{{{0.1f, 0.1f, 0.1f, 1.0f}}}},
		}}},
		.depthAttachment  = {
			 .imageView   = depth_image.GetView(),
			 .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			 .loadOp      = vk::AttachmentLoadOp::eClear,
			 .storeOp     = vk::AttachmentStoreOp::eDontCare,
			 .clearValue  = {{{{1.0f, 0}}}},
        },
	});
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	camera.updateProjectionViewInverse();
	BRDFConstants constants{
		.view_proj = camera.getProjViewInv(),
		.material  = {
			 .base_color = {0.8f, 0.8f, 0.8f, 0.8f},
			 .metallic   = 0.0f,
			 .roughness  = 0.5f,
        },
		.light = {
			.position          = vec3(2.0, 2.0, 2.0),
			.range             = 10.0,
			.color             = vec3(0.8, 0.8, 0.8),
			.intensity         = 8.0,
			.ambient_color     = vec3(0.9, 0.9, 0.9),
			.ambient_intensity = 0.03,
		},
		.camera_pos = camera.getPosition(),
		.pad        = static_cast<int>(weights_offset),
	};

	// Update weight offsets in push constants
	for (int i = 0; i < kNetworkLinearLayers; ++i) {
		auto layer = network.GetLayer<Linear>(i);
		auto offset_base = this->weights_offset;
		constants.weights_offsets[i] = offset_base + layer.GetWeightsOffset();
		constants.bias_offsets[i]    = offset_base + layer.GetBiasesOffset();
		// std::printf("Layer %d weights_offset %d bias_offset %d\n", i, constants.weights_offsets[i], constants.bias_offsets[i]);
	}

	u32 const constants_offset = 0u;
	cmd.pushConstants(
		pipeline_layout,
		vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		constants_offset, sizeof(constants), &constants);

	// std::printf("camera pos %f %f %f\n", constants.camera_pos.x, constants.camera_pos.y, constants.camera_pos.z);

	// u32 vertex_count = GetCubeVertices().size();
	u32 vertex_count = sphere.GetVertexCount();
	cmd.bindVertexBuffers(0, device_buffer, vertices_offset);
	cmd.bindIndexBuffer(device_buffer, indices_offset, vk::IndexType::eUint32);
	// cmd.draw(3, 1, 0, 0);
	// cmd.draw(vertex_count, 1, 0, 0);
	cmd.drawIndexed(sphere.GetIndexCount(), 1, 0, 0, 0);
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

void BRDFSample::RecreateSwapchain(int width, int height) {
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}
static auto last_frame_time = std::chrono::steady_clock::now();

void BRDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		DrawWindow(pipelines[u32(BrdfFunctionType::eDefault)], linear_network);
	} while (true);
}

auto BRDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	for (std::string_view const arg : std::span(argv + 1, argc - 1)) {
		if (arg == "--test" || arg == "-t") bIsTestMode = true;
		else if (arg == "--verbose") verbose = true;
		else if (arg == "--validation") bUseValidation = true;
		else return arg.data();
	}
	return nullptr;
}

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	BRDFSample sample;

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
