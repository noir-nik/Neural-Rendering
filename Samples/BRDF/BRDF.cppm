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

import FastKan;

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

using FastKanOffsets = std::vector<FastKanLayerBase<u64>>;

struct SpecData {
	BrdfFunctionType function_type = BrdfFunctionType::eCoopVec;
	u32              function_id   = 0; // inline
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
	auto CreateSkyboxPipeline(vk::ShaderModule shader_module) -> vk::Pipeline ;

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

	bool is_test_mode = false;
	bool verbose      = false;
	// bool use_validation = true;
	bool use_validation = false;

	BrdfFunctionType function_type = BrdfFunctionType::eCoopVec;
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

	Image accumulator_image;

	// auto hasattr0(auto proj) -> bool { return !std::invoke(proj, *this).empty(); }
	auto hasattr(std::string_view BRDFSample::* arg) -> bool { return !(this->*arg).empty(); };

	bool benchmark_single = false;

	std::optional<u32> function_id = std::nullopt;
	// u32 function_id = 0;

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

	static constexpr int kTestFunctionsCount = 50;

	std::array<vk::Pipeline, kTestFunctionsCount> pipelines_header = {};

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

	Camera camera{{
		// .position = {1.0f, -1.0f, 2.0f},
		.position = float3{1.0f, 1.0f, 2.5f},// * 1.8,
		// .position = float3{-0.82666, -0.10470,  0.55289}*3,
		// .position = {-0.05682, 0.55289, 1.63774,}
		.fov      = 50.0f,
		.z_near   = 0.01f,
		.z_far    = 100.0f,
	}};

	Vertex vv{};
};
