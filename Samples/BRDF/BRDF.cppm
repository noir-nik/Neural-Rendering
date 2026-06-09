module;

// #include <cstddef>

// #include <cstdint>
#include "Log.h"

export module BRDFSample;

import NeuralGraphics;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import WeightsLoader;
import SamplesCommon;
import VulkanRHI;
import Math;
import Vdevice;
import vulkan;
import std;

import FastKANCoopVec;

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t

// using namespace Utils;
using namespace math;

#include "Shaders/BRDFConstants.h"

using mesh::UVSphere;
using VulkanRHI::Buffer;
using VulkanRHI::Image;
using u32 = std::uint32_t;

struct TestOptions {
	int2 resolution = {640, 480};
	// TestType    test_type    = TestType::eSDF;
	// NetworkType network_type = NetworkType::eScalarInline;
	int test_count = 1;
};

struct SpecData {
	BrdfFunctionType function_type = BrdfFunctionType::eCoopVec;
	u32              function_id   = 0; // inline
};

#if defined(WITH_UI) && WITH_UI
extern "C++" struct ImDrawData;
#endif

template <typename T>
using CSpan = std::span<T const>;
template <typename T>
using Span = std::span<T>;

using SV = std::string_view;

struct BRDFModelData {
	std::string_view name;
	std::string_view type;

	u32 layers;
	u32 layer_size;
	u32 grids;
	u32 learnable_params;
	u32 total_params;
};

class BRDFSample {
public:
	static constexpr u32 kFramesInFlight = 3;

	using Vertex = mesh::Vertex;

	~BRDFSample();

	void Init();
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

	void CreatePipelines();

	auto with_coop_vec() -> bool { return function_type == BrdfFunctionType::eCoopVec; }

	[[nodiscard]]
	auto CreatePipeline(vk::ShaderModule vertex_shader_module, SpecData const& info = {}) -> vk::Pipeline;
	[[nodiscard]]
	auto CreateSkyboxPipelinePrivate(vk::ShaderModule shader_module) -> vk::Pipeline;
	auto CreateSkyboxPipeline(vk::ShaderModule shader_module, SpecData const& info) -> vk::Pipeline;

	// void BuildNetwork();
	struct NetworkBufferInfo {
		std::string_view file_name;
		std::string_view header;
	};
	void extracted();
	void CreateAndUploadBuffers(NetworkBufferInfo const& info);
	// void ReadKANWeights(NetworkBufferInfo const& info);

	// Return time in nanoseconds
	auto GetQueryResult() -> u64;
	void RecreateSwapchain(int width, int height);
	void SaveSwapchainImageToFile(std::string_view filename);

	auto GetAllocator() const -> vk::AllocationCallbacks const* { return allocator; }
	auto GetPipelineCache() const -> vk::PipelineCache { return pipeline_cache; }

	bool IsTestMode() const { return is_test_mode; }

	auto ParseArgs(int argc, char const* argv[]) -> char const*;

#if defined(WITH_UI) && WITH_UI
	void CreateImGui();
	void DrawImGui(vk::CommandBuffer cmd, ImDrawData* imDrawData);
	void ImGuiNewFrame();
	void ImGuiShutdown();
	void DrawUI();

	vk::DescriptorPool imgui_descriptor_pool;
	bool               is_ui_visible     = true;
	bool               is_models_visible = true;
	bool               is_fps_visible    = true;
#endif

	bool is_test_mode = false;
	bool verbose      = false;
	// bool use_validation = true;
	bool use_validation = false;

	BrdfFunctionType function_type = BrdfFunctionType::eClassic;
	// std::optional<BrdfFunctionType> function_type = std::nullopt;
	std::string_view weights_file_name;
	std::string_view kan_weights_file_name;

	std::string_view cubemap_folder_path;
	std::string_view obj_path;

	u32 num_vertices = 0;
	u32 num_indices  = 0;

	static constexpr auto kCubeSideCount = 6;

	// std::array<Image, kCubeSideCount> cubemap_images;
	Image       cubemap_image;
	vk::Sampler cubemap_sampler{};

	bool is_cubemap_visible_ = true;
	void set_cubemap_visible(bool value) {
		is_cubemap_visible_ = value;
	}
	bool is_cubemap_visible() {
		return is_cubemap_visible_;
	}
	bool is_cubemap_loaded_ = false;
	auto is_cubemap_loaded() -> bool {
		return is_cubemap_loaded_;
	}

	static constexpr auto kMinFastKANVersion = 0;
	static constexpr auto kMaxFastKANVersion = 3;
	int                   fastkan_version    = 0;

	Image accumulator_image;

	// auto hasattr0(auto proj) -> bool { return !std::invoke(proj, *this).empty(); }
	auto hasattr(std::string_view BRDFSample::* arg) -> bool { return !(this->*arg).empty(); };

	bool benchmark_single = false;

	std::optional<u32> function_id = std::nullopt;
	// u32 function_id = 0;

	using Duration = ::std::chrono::steady_clock::duration;
	Duration elapsed_cpu;
	float    elapsed_last_frame_ms = 1e-3f;

	GLFWWindow window{};

	Mouse mouse;

	void ProcessViewportInput();

	vk::Instance                    instance;
	vk::AllocationCallbacks const*  allocator     = nullptr;
	VmaAllocator                    vma_allocator = nullptr;
	vk::PipelineCache               pipeline_cache;
	vk::DebugUtilsMessengerEXT      debug_messenger;
	std::span<char const* const>    enabled_layers;
	std::vector<vk::PhysicalDevice> vulkan_physical_devices;
	PhysicalDevice                  physical_device;
	// vk::Device                      device;
	VDevice              device;
	VulkanRHI::Swapchain swapchain;
	bool                 swapchain_dirty = false;
	vk::SurfaceKHR       surface;

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

	std::array<vk::Pipeline, u32(BrdfFunctionType::eCount)> pipelines = {};
	vk::Pipeline                                            skybox_pipeline{};

	static constexpr u32 kPipelineFallbackCount = 2;

	std::array<vk::Pipeline, kPipelineFallbackCount> pipelines_fallback = {};

	// 	static constexpr int _HeaderNames_count_array[] = {
	// #define BRDF_NAME(x) 0,
	// #include "HeaderNames.def"
	// 	};

	// static constexpr int kTestFunctionsCount = std::size(_HeaderNames_count_array);

	// std::array<vk::Pipeline, kTestFunctionsCount> pipelines_header = {};

	// std::vector<vk::Pipeline> pipelines_header = {};

	static constexpr auto kMaxGeneratedPipelines = std::size_t{GENERATED_MODELS_COUNT * 2};

	std::array<vk::Pipeline, kMaxGeneratedPipelines> generated_pipelines{};

	struct PipelineData {

		static constexpr auto kUndefined = static_cast<u32>(-1);

		Utils::BinDataOption code{};

		u32 pipeline_id{kUndefined};

		auto GetPipeline(u32 id) -> vk::Pipeline;
	};

	std::array<PipelineData, GENERATED_MODELS_COUNT * 2> generated_data{};

	auto GeneratedPipeline(u32 i) -> vk::Pipeline;
	auto EnsurePipeline(u32 i) -> vk::Pipeline;

	auto GeneratedNames() -> CSpan<BRDFModelData>;

	// vk::Pipeline getpipeline

	Buffer         device_buffer;
	vk::DeviceSize vertices_offset = 0;
	vk::DeviceSize indices_offset  = 0;

	// vk::DeviceSize linear_weights_offset     = 0;
	// vk::DeviceSize linear_weights_offset_f16 = 0;
	// vk::DeviceSize optimal_weights_offset    = 0;

	std::array<vk::DeviceSize, u32(BrdfFunctionType::eCount)> weights_offsets;

	FastKanOffsets kan_offsets;

	Buffer staging_buffer;

	UVSphere sphere = UVSphere(1.0f, 32, 16);

	Image depth_image;
	bool  use_depth = true;

	std::string_view header = "";

	VulkanCoopVecNetwork networks[u32(BrdfFunctionType::eCount)];

	void RecordCommands(vk::Pipeline pipeline);
	auto DrawWindow(vk::Pipeline pipeline) -> u64;
	auto DrawWindow() -> u64;

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

	Camera camera{{}};

	Vertex vv{};
};
