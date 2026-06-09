module;
#include <cassert> // assert

module BRDFSample;

// #include <cstddef>

#include "CheckResult.h"
#include "Log.h"
#include "Shaders/BRDFConfig.h"

import NeuralGraphics;
import vulkan;
import WindowManager;
import VulkanExtensions;
import VulkanFunctions;
import Window;
import vk_mem_alloc;
import SamplesCommon;
import Math;
import std;

#if defined(WITH_UI) && WITH_UI
import imgui;
#endif

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
	if (function_id) {
		// std::printf("DrawWindow(pipelines_header[%u])\n", u32(*function_id));
		return DrawWindow(pipelines_header[u32(*function_id)]);
	} else {
		// std::printf("DrawWindow(pipelines[%u])\n", u32(function_type));

		auto const pipeline_num =
			hasattr(&BRDFSample::cubemap_folder_path)
				? 1
				: 0;
		return DrawWindow(pipelines_fallback[pipeline_num]);
		// return DrawWindow(pipelines[u32(function_type)]);
	}
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
static u32 factor       = 3;
static u32 period_rot   = 120 * factor;
static u32 period_pause = 180 * factor;

static u32    frame_count{};
static u32    start_frame{static_cast<u32>(60 * 5 * factor)};
static float3 prev_camera_pos = {0.0f, 0.0f, 0.0f};

void BRDFSample::RecordCommands(vk::Pipeline pipeline) {
	LOG_DEBUG("BRDFSample::RecordCommands()");

	// bool movesine = 1;
	bool movesine =
		0
		// 1
		//
		;
	// bool movesine = ;
	// std::printf("frame_count: %d\n", frame_count);

	static u32 rot   = 0;
	static u32 pause = 0;

	auto const pi = 3.1415926535897932384626433832795f;
	if (movesine) {

		int diff = frame_count - start_frame;

		static u32 counter = 0;
		if (rot < period_rot) {
			++rot;
			// rot = 120;
			if (rot == (period_rot / 2) && diff > 0) {
				counter++;
			}
		} else {
			if (pause == period_pause) {
				pause = 0;
				rot   = 0;
			}

			++pause;
		}

		auto t    = float(rot) / period_rot * pi;
		auto sint = std::sinf(t);
		auto l    = std::lerp(0.f, pi, sint);

		auto dm = std::fmaxf(sint, 0.0f);

		auto delta_x = (dm);

		*function_id = counter % this->pipelines_header.size();

		// std::printf("t: %f, sin: %f, l: %f\n", t, dm, l);
		// std::printf("diff: %u, function_id: %u\n", diff, u32(*function_id));

		float tw = float(120) / period_rot;

		// tw = 1.f;
		// tw = rt;

		float2 delta_pos = {delta_x * 1.035f * 0.5f * 0.5f * tw / pi, 0.f};

		// camera.moveWithCursor(width, height, delta_x, delta_y);

		if (frame_count > start_frame) {
			auto&       camera_pos     = camera.getPosition();
			auto const& camera_right   = camera.getRight();
			auto const& camera_up      = camera.getUp();
			auto const& camera_forward = camera.getForward();
			// all float3
			camera_pos -= camera.focus;

			// Correct upside down
			float3 world_up = float3(0.0f, 1.0f, 0.0f);
			// float rotation_sign = dot(camera_up, world_up) < 0.0f ? -1.0f : 1.0f;
			float rotation_sign = 1.0f;

			camera.view =
				camera.view
					// .rotate(camera_right, -delta_pos.y * camera.rotation_factor)
					//   | rotate(camera_up, rotation_sign * delta_pos.x * camera.rotation_factor);
					.rotate(world_up, rotation_sign * delta_pos.x);
			// .rotate(world_up, rotation_sign * delta_pos.x);
			camera_pos += camera.focus;
		}
	}

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
			// .clearValue = {{{{0.2f, 0.2f, 0.2f, 0.0f}}}},
			.clearValue = {{{{0.1f, 0.1f, 0.1f, 0.0f}}}},
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

	// camera.getForward() *= -1.0;
	// camera.updateProjectionViewInverse();
	// camera.getForward() *= -1.0;

	static auto old_f_id = 0;
	auto const  f_id     = function_id.value_or(0);

	int reset_accumulation =
		((length(prev_camera_pos - camera.getPosition())) > 0.001f)
		|| (old_f_id != f_id);

	old_f_id = f_id;

	prev_camera_pos = camera.getPosition();
	// std::printf("prev_camera_pos: %f, %f, %f\n", prev_camera_pos.x, prev_camera_pos.y, prev_camera_pos.z);
	// std::printf("camera pos     : %f, %f, %f\n", camera.getPosition().x, camera.getPosition().y, camera.getPosition().z);
	// std::printf("reset_accumulation: %d\n", reset_accumulation);
	// std::printf("frame_count: %d\n", frame_count);

	auto view = camera.view;
	auto proj = camera.proj;

	auto view_proj_base = proj * view.affineInverse();
	auto view_proj      = OpenglToVulkanProjectionMatrixFix() * view_proj_base;

	float4 test = {vv.pos[0], vv.pos[1], vv.pos[2], 1.0f};

	auto res = view_proj * test;

	if (0) {
		PrintMat4(view_proj);
		std::printf("Result: %f, %f, %f, %f\n", res.x, res.y, res.z, res.w);
	}
	auto light_pos = (vec3(1.2, 1.2, 1.2) * 1.2);
	light_pos      = rotate(light_pos, {0, 1, 0}, 180 * math::DEG_TO_RAD);

	BRDFConstants constants{
		// .view_proj = camera.getProjViewInv(),
		.view_proj = view_proj,
		.material  = {
			.base_color = float4{0.8, 0.8, 0.8, 1.0},
			.metallic   = 0.5f,
			.roughness  = 0.5f,
		},

		.light = {
			.position          = light_pos,
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
	auto bind_push = [&](vk::Pipeline pipeline) {
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

		u32 const constants_offset = 0u;
		cmd.pushConstants(
			pipeline_layout,
			vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
			constants_offset, sizeof(constants), &constants);
	};

	// Skybox
	if (is_cubemap_loaded() && is_cubemap_visible()) {
		auto ffix = [] {
			float4x4 res;
			res[1][1] = -1.0f;
			// res[2][2] = 0.5f;
			// res[2][3] = 0.5f;
			return res;
			// return res;
		};

		constants.view_proj = (ffix() * view_proj_base).inverse();

		bind_push(skybox_pipeline);

		// u32 vertex_count = num_vertices;
		u32 vertex_count = 3;

		cmd.draw(vertex_count, 1, 0, 0);
	}
	// Model
	{ // auto pipeline = pipelines[u32(function_type)];

		constants.view_proj = OpenglToVulkanProjectionMatrixFix() * (view_proj_base);

		bind_push(pipeline);

		u32 vertex_count = num_vertices;
		u32 index_count  = num_indices;
		cmd.bindVertexBuffers(0, device_buffer, vertices_offset);
		cmd.bindIndexBuffer(device_buffer, indices_offset, vk::IndexType::eUint32);
		// cmd.draw(3, 1, 0, 0);
		// cmd.draw(vertex_count, 1, 0, 0);

		cmd.drawIndexed(index_count, 1, 0, 0, 0);
	}
#if defined(WITH_UI) && WITH_UI
	if (is_ui_visible) {
		ImGuiNewFrame();

		// ImGui::ShowDemoWindow();
		DrawUI();

		ImGui::Render();
		auto imdata = ImGui::GetDrawData();
		DrawImGui(cmd, imdata);
	}
#endif

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

auto fix_framerate() -> BRDFSample::Duration {
	auto const  max_fps   = 144.0f;
	static auto last_time = std::chrono::steady_clock::now();
	auto const  now       = std::chrono::steady_clock::now();
	auto const  elapsed   = now - last_time;
	// this->elapsed_cpu = now - last_time;

	last_time = now;
	std::this_thread::sleep_for(std::chrono::duration<float, std::milli>(1000.0f / max_fps) - elapsed);

	return elapsed;
}

void BRDFSample::Run() {
	LOG_DEBUG("BRDFSample::Run()");

	u32 const end_frame{static_cast<u32>(start_frame + (period_pause + period_rot) * pipelines_header.size() - 10)};

	// current_fps = 1000.f / elapsed_ms;
	do {
		// WindowManager::WaitEvents();
		WindowManager::PollEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		u64 elapsed_ns = DrawWindow();

		this->elapsed_last_frame_ms = elapsed_ns / 1000000.0;
		verbose&& std::printf("%f ms, fps: %f\n", this->elapsed_last_frame_ms, 1000.f / this->elapsed_last_frame_ms);

		this->elapsed_cpu = fix_framerate();

		if (frame_count > end_frame) {
			return;
		}
	} while (true);
}
