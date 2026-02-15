module;
#include <cassert> // assert

module BRDFSample;

// #include <cstddef>

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

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t

using namespace Utils;

using VulkanRHI::Buffer;
using VulkanRHI::Image;

static_assert(sizeof(BRDFConstants) <= 256);

void PrintMat4(float4x4 const& mat, int precision = 5, int digits = 2) {

	auto d = digits + precision + 1;

	auto print_elem = [&](float elem) { std::printf("%*.*f", d, precision, elem); };
	for (int i = 0; i < 4; ++i) {
		std::printf("%*.*f, %*.*f, %*.*f, %*.*f\n", d, precision, mat[i][0], d, precision, mat[i][1], d, precision, mat[i][2], d, precision, mat[i][3]);
	}
}

auto BRDFSample::GetQueryResult() -> u64 {
	LOG_DEBUG("BRDFSample::GetQueryResult()");
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

auto BRDFSample::DrawWindow() -> u64 {
	LOG_DEBUG("BRDFSample::DrawWindow()");
	if (function_id)
		return DrawWindow(pipelines_header[u32(*function_id)]);
	else
		return DrawWindow(pipelines[u32(function_type)]);
};

auto BRDFSample::DrawWindow(vk::Pipeline pipeline) -> u64 {
	LOG_DEBUG("BRDFSample::DrawWindow()");
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
	CHECK_VULKAN_RESULT(device.resetCommandPool(swapchain.GetCurrentCommandPool()));
	if (!HandleSwapchainResult(swapchain.AcquireNextImage())) return 0ull;
	CHECK_VULKAN_RESULT(device.resetFences(1, &swapchain.GetCurrentFence()));
	RecordCommands(pipeline);
	if (!HandleSwapchainResult(swapchain.SubmitAndPresent(queue, queue))) return 0ull;

	u64 elapsed = GetQueryResult();
	swapchain.EndFrame();
	return elapsed;
}
static u32    frame_count{};
static float3 prev_camera_pos = {0.0f, 0.0f, 0.0f};

void BRDFSample::RecordCommands(vk::Pipeline pipeline) {
	LOG_DEBUG("BRDFSample::RecordCommands()");
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	// auto depth_extent = depth_image.GetExtent();
	// if (static_cast<u32>(width) > depth_extent.width || static_cast<u32>(height) > depth_extent.height) {
	// 	depth_image.Recreate({static_cast<u32>(width), static_cast<u32>(height), 1});
	// }

	for (auto i : {&depth_image, &accumulator_image}) {
		if (static_cast<u32>(width) > i->GetExtent().width || static_cast<u32>(height) > i->GetExtent().height) {
			i->Recreate({static_cast<u32>(width), static_cast<u32>(height), 1});
		}
	}

	vk::Rect2D               render_rect{{0, 0}, {static_cast<u32>(width), static_cast<u32>(height)}};
	VulkanRHI::CommandBuffer cmd = swapchain.GetCurrentCommandBuffer();
	CHECK_VULKAN_RESULT(cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
	cmd.resetQueryPool(timestamp_query_pool, GetCurrentTimestampIndex(), kTimestampsPerFrame);
	cmd.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timestamp_query_pool, GetCurrentTimestampIndex());

	vk::Image swapchain_image = swapchain.GetCurrentImage();
	// cmd.SetViewport({0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f, 1.0f});
	cmd.SetViewport({0.0f, static_cast<float>(height), static_cast<float>(width), -static_cast<float>(height), 1.0f, 0.0f});
	cmd.SetScissor(render_rect);
	cmd.Barrier2({
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
			// .clearValue  = {{{{0.5f, 0.5f, 0.5f, 0.0f}}}},
			.clearValue = {{{{0.2f, 0.2f, 0.2f, 0.0f}}}},
			// .clearValue = {{{{1.f, 1.f, 1.f, 1.0f}}}},
			// .clearValue  = {{{{0.f, 0.f, 0.f, 1.0f}}}},
		}}},
		.depthAttachment  = {
			 .imageView   = depth_image.GetView(),
			 .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			 .loadOp      = vk::AttachmentLoadOp::eClear,
			 .storeOp     = vk::AttachmentStoreOp::eStore,
			 .clearValue  = {{{{1.0f, 0}}}},
        },
	});
	// auto pipeline = pipelines[u32(function_type)];
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	camera.getForward() *= -1.0;
	camera.updateProjectionViewInverse();
	camera.getForward() *= -1.0;

	int reset_accumulation = (length(prev_camera_pos - camera.getPosition())) > 0.001f;
	// reset_accumulation     = 1;

	prev_camera_pos = camera.getPosition();
	// std::printf("prev_camera_pos: %f, %f, %f\n", prev_camera_pos.x, prev_camera_pos.y, prev_camera_pos.z);
	// std::printf("camera pos     : %f, %f, %f\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
	// std::printf("reset_accumulation: %d\n", reset_accumulation);
	// std::printf("frame_count: %d\n", frame_count);

	BRDFConstants constants{
		.view_proj = camera.getProjViewInv(),
		.material  = {
			 .base_color = float4{0.8, 0.8, 0.8, 1.0},
			 .metallic   = 0.5f,
			 .roughness  = 0.5f,
        },
		.light = {
			.position          = vec3(1.2, 1.2, 1.2),
			.range             = 10.0,
			.color             = vec3(0.75, 0.75, 0.75),
			.intensity         = 8.0,
			.ambient_color     = vec3(0.9, 0.9, 0.9),
			.ambient_intensity = 0.03,
		},
		.camera_pos = camera.getPosition(),

		.frame_count        = frame_count++,
		.reset_accumulation = reset_accumulation,
	};

	// PrintMat4(camera.getView());
	// std::printf("\n");

	// fastkan
	// ENABLE_FAST_KAN
	/*
	constants.fast_kan.num_layers = kan_offsets.size();
	for (int i = 0; i < kan_offsets.size(); ++i) {
		// FastKanLayerBufferOffsets c
		constants.fast_kan.offsets[i].rbf_grid      = kan_offsets[i].rbf_grid();
		constants.fast_kan.offsets[i].spline_weight = kan_offsets[i].spline_weight();
		constants.fast_kan.offsets[i].base_weight   = kan_offsets[i].base_weight();
		constants.fast_kan.offsets[i].base_bias     = kan_offsets[i].base_bias();
	}
 */
	// Update weight offsets in push constants
	auto network = networks[u32(function_type)];
	for (int i = 0; i < network.GetLayers().size(); ++i) {
		auto layer       = network.GetLayer<Linear>(i);
		auto offset_base = weights_offsets[u32(function_type)];

		// ENABLE_MLP
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

	// u32 vertex_count = sphere.GetVertexCount();
	// u32 index_count  = sphere.GetIndexCount();
	u32 vertex_count = num_vertices;
	u32 index_count  = num_indices;
	cmd.bindVertexBuffers(0, device_buffer, vertices_offset);
	cmd.bindIndexBuffer(device_buffer, indices_offset, vk::IndexType::eUint32);
	// cmd.draw(3, 1, 0, 0);
	// cmd.draw(vertex_count, 1, 0, 0);
	
	cmd.drawIndexed(index_count, 1, 0, 0, 0);
	cmd.endRendering();
	cmd.Barrier2({
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
	LOG_DEBUG("BRDFSample::RecreateSwapchain()");
	for (auto& frame : swapchain.GetFrameData()) {
		CHECK_VULKAN_RESULT(device.waitForFences(1, &frame.GetFence(), vk::True, std::numeric_limits<u32>::max()));
	}
	CHECK_VULKAN_RESULT(swapchain.Recreate(width, height));
	swapchain_dirty = false;
}

void BRDFSample::SaveSwapchainImageToFile(std::string_view filename) {
	LOG_DEBUG("BRDFSample::SaveSwapchainImageToFile()");
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

	cmd.Barrier2({
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
	// pending_image_save = false;
}

void fix_framerate() {
	auto        fps       = 60.0f;
	static auto last_time = std::chrono::steady_clock::now();
	auto const  now       = std::chrono::steady_clock::now();
	auto const  elapsed   = now - last_time;
	last_time             = now;
	std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.0f / fps) - elapsed);
}

void BRDFSample::Run() {
	LOG_DEBUG("BRDFSample::Run()");
	do {
		// WindowManager::WaitEvents();
		WindowManager::PollEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		u64       elapsed_ns = DrawWindow();
		verbose&& std::printf("%f ms\n", elapsed_ns / 1000000.0);
		fix_framerate();
	} while (true);
}

template <typename Range, typename Proj = std::identity>
constexpr inline auto contains(Range&& range, auto&& value, Proj&& proj = std::identity{}) {
	for (auto&& v : range)
		if (std::invoke(proj, v) == value)
			return true;
	return false;
};

void BRDFSample::RunBenchmark(TestOptions const& options) {
	LOG_DEBUG("BRDFSample::RunBenchmark()");
	struct TestData {
		vk::Pipeline                pipeline;
		VulkanCoopVecNetwork const* network;
	};
	// WindowManager::WaitEvents();
	// if (window.GetShouldClose()) return;

	// window.SetWindowMode(WindowMode::eFullscreen);
	auto [width, height] = options.resolution;
	window.SetSize(width, height);
	// window.Hide();
	// std::printf("Resizing to %dx%d\n", width, height);
	RecreateSwapchain(width, height);
	constexpr u32 kTestRunsCount = 64;

	// constexpr u32 kMaxTestKinds = std::to_underlying(BrdfFunctionType::eCount);
	constexpr u32 kMaxTestKinds = kTestFunctionsCount;

	constexpr u32 kMaxTests = 64;
	// std::vector<std::array<u64, kMaxTestKinds>> test_times(kTestRunsCount);
	std::array<std::array<u64, kMaxTests>, kTestRunsCount> test_times;

	int first_test{}, last_test = kMaxTestKinds;

	if (benchmark_single) {
		first_test = std::to_underlying(function_type);
		last_test  = first_test + 1;
	} else {
		first_test = *function_id;
		last_test  = first_test + 1;
	}

	bool is_header = true;
	// bool is_header = false;
	if (is_header) {
		first_test = 0;
		last_test  = kTestFunctionsCount;
		// last_test  = 5;
	}

	// std::printf("Running %d tests\n", last_test - first_test);
	// std::printf("test id: %d\n", first_test);

	// BrdfFunctionType skip[] = {BrdfFunctionType::eWeightsInHeader};
	BrdfFunctionType skip[] = {};

	// std::mem_fn(&BRDFSample::DrawWindow);
	LOG_DEBUG("BRDFSample::DrawWindow()");

	auto draw = [&](u32 id) {
		if (is_header) {
			return DrawWindow(pipelines_header[id]);
		} else {
			return DrawWindow(pipelines[id]);
		};
	};

	// Warm up gpu clocks
	constexpr u32 kWarmupCount = 2;
	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kWarmupCount; ++iter) {
			(void)draw(t_i);
		}
	}

	bool with_in_header = false;

	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
			// WindowManager::PollEvents();
			u64   time_nanoseconds = draw(t_i);
			float ns_per_tick      = physical_device.GetNsPerTick();
			float elapsed_ms       = (time_nanoseconds * ns_per_tick) / 1e6f;
			test_times[iter][t_i]  = time_nanoseconds;
		}
	}

	char const* names[] = {"Classic", "CoopVec", "WeightsInBuffer", "WeightsInBufferFloat16", "WeightsInHeader", "Kan"};

	// Print csv
	// std::printf("Print csv\n");
	// std::printf("Classic,CoopVec,WeightsInBuffer,WeightsInBufferFloat16,WeightsInHeader\n");

	char const* header_names[] = {
#define BRDF_NAME(x) #x,
// #include "SINEKAN_HeaderNames.def"
// #include "FASTKAN_HeaderNames.def"
#include "CHEBYKAN_HeaderNames.def"
		// #include "RELUKAN_HeaderNames.def"
	};

	for (u32 t_i = first_test; t_i < last_test; ++t_i) {
		if (contains(skip, BrdfFunctionType(t_i))) continue;
		if (is_header) {
			std::printf("%s", header_names[t_i]);
		} else {
			// std::printf("t_i %u", t_i);
			std::printf("%s", names[t_i]);
		}
		if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
	}
	std::printf("\n");

	// std::printf("Print times\n");
	for (u32 iter = 0; iter < kTestRunsCount; ++iter) {
		auto const& tests_row = test_times[iter];
		// print with ,
		for (u32 t_i = first_test; t_i < last_test; ++t_i) {
			if (contains(skip, BrdfFunctionType(t_i))) continue;
			std::printf("%llu", tests_row[t_i]);
			if (t_i < last_test - 1 && !contains(skip, BrdfFunctionType(t_i + 1))) std::printf(",");
		}
		std::printf("\n");
	}
}

auto BRDFSample::ParseArgs(int argc, char const* argv[]) -> char const* {
	LOG_DEBUG("BRDFSample::ParseArgs()");
	auto args_range = std::span(argv + 1, argc - 1);

	if (std::ranges::contains(args_range, std::string_view("--help")))
		return "--help";

	for (auto it = args_range.begin(); it != args_range.end(); ++it) {
		auto arg = std::string_view(*it);
		if (arg == "--benchmark" || arg == "-b") is_test_mode = true;
		else if (arg == "--verbose" || arg == "-v") verbose = true;
		else if (arg == "--validation") use_validation = true;
		else if (arg == "--kind") {
			if ((it + 1) == args_range.end()) return "expected <kind>";
			auto kind = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(kind.data(), kind.data() + kind.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= std::to_underlying(BrdfFunctionType::eCount)) return *(it + 1);
			function_type    = static_cast<BrdfFunctionType>(value);
			benchmark_single = true;
			++it;
		} else if (arg == "-f") {
			if ((it + 1) == args_range.end()) return "expected <id>";
			auto str = std::string_view(*(it + 1));
			int  value;
			if (std::from_chars(str.data(), str.data() + str.size(), value).ec != std::errc()) return *(it + 1);
			if (value < 0 || value >= kTestFunctionsCount) return *(it + 1);
			// function_type    = BrdfFunctionType::eWeightsInHeader;
			benchmark_single = true;
			function_id      = value;
			++it;
		} else if (arg == "-w") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));
			// if (!std::filesystem::exists(str)) return *(it + 1);
			weights_file_name = str;
			++it;
		} else if (arg == "-kw") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));

			kan_weights_file_name = str;
			++it;
		} else if (arg == "-cm") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <folder>";
			auto str = std::string_view(*(it + 1));

			cubemap_folder_path = str;
			++it;
		} else if (arg == "-obj") { // input weights file
			if ((it + 1) == args_range.end()) return "expected <file>";
			auto str = std::string_view(*(it + 1));

			obj_path = str;
			++it;
		} else return *it;
	}

	return nullptr;
}

auto PrintUsage([[maybe_unused]] int argc, char const* argv[]) -> void {
	std::printf("Usage: %s [--help] [--benchmark | -b] [--verbose | -v] [--validation] [--kind <kind>]\n",
				std::filesystem::path(argv[0]).filename().string().c_str());
	std::printf("  --kind <kind>\n");
	std::printf("      Kind of BRDF function to run:\n");
	std::printf("        0: Classic\n");
	std::printf("        1: Coop Vec (Default)\n");
	std::printf("        2: Weights in buffer\n");
	std::printf("        3: Weights in buffer float16\n");
	std::printf("        4: Weights in header\n");
	std::printf("        5: Kan\n");
	std::printf("  --benchmark | -b\n");
	std::printf("      Run benchmark\n");
	std::printf("  --verbose | -v\n");
	std::printf("      Run benchmark with verbose output\n");
	std::printf("  --validation\n");
	std::printf("      Enable validation\n");
	std::printf("\n");
};

auto main(int argc, char const* argv[]) -> int {
	std::filesystem::current_path(std::filesystem::absolute(argv[0]).parent_path());
	BRDFSample sample;

	if (char const* unknown_arg = sample.ParseArgs(argc, argv); unknown_arg) {
		if (unknown_arg != std::string_view("--help"))
			std::printf("Error in argument: %s\n", unknown_arg);
		PrintUsage(argc, argv);
		return 0;
	}

	TestOptions options{
		.resolution = {640, 480},
		.test_count = 64,
	};

	int2 res_arr[] = {
		{1920, 1080},
		// {3840, 2160},
		// {512, 512},
		{640, 480},
		{1280, 720},
		{1920, 1080},
		{2560, 1440},
		// {3840, 2160},
	};

	auto res_count = //

		// std::size(res_arr);
		1;

	sample.Init();
	if (sample.IsTestMode()) {
		for (int i = 0; i < res_count; ++i) {
			options.resolution = res_arr[i];
			// std::printf("resolution: %d x %d\n", res_arr[i].x, res_arr[i].y);
			sample.RunBenchmark(options);
		}
	} else {
		sample.Run();
	}
	return 0;
}
