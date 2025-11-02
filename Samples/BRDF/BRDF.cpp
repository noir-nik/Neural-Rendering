module;

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
import WeightsLoader;
import SamplesCommon;
import Math;
import std;

using numeric::float16_t;

#ifdef COOPVEC_TYPE
#undef COOPVEC_TYPE
#endif
#define COOPVEC_TYPE numeric::float16_t

using namespace Utils;

using VulkanRHI::Buffer;
using VulkanRHI::Image;

void PrintMat4(float4x4 const& mat, int precision = 5, int digits = 2) {

	auto d = digits + precision + 1;

	auto print_elem = [&](float elem) { std::printf("%*.*f", d, precision, elem); };
	for (int i = 0; i < 4; ++i) {
		std::printf("%*.*f, %*.*f, %*.*f, %*.*f\n", d, precision, mat[i][0], d, precision, mat[i][1], d, precision, mat[i][2], d, precision, mat[i][3]);
	}
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

template <typename T>
	requires(std::is_same_v<T, float> || std::is_same_v<T, float16_t>)
auto PrintLayerBiases(Linear layer, std::byte const* parameters) -> void {
	if (!parameters) return;
	auto const* p_bias = reinterpret_cast<T const*>(parameters + layer.GetBiasesOffset());

	for (u32 i = 0; i < layer.GetBiasesCount(); ++i) {
		std::printf("%10.6f ", float(p_bias[i]));
	}
	std::printf("\n");
}

void BRDFSample::CreateAndUploadBuffers(NetworkBufferInfo const& network_info) {
	sphere                            = UVSphere(1.0f, 32 * 2, 16 * 2);
	std::size_t vertices_size_bytes   = sphere.GetVertexCount() * sizeof(UVSphere::Vertex);
	std::size_t indices_size_bytes    = sphere.GetIndexCount() * sizeof(UVSphere::IndexType);
	std::size_t alignment             = sizeof(float) * 4;
	std::size_t vertices_size_aligned = AlignUpPowerOfTwo(vertices_size_bytes, alignment);
	std::size_t indices_size_aligned  = AlignUpPowerOfTwo(indices_size_bytes, alignment);

	// auto weights_path = "Assets/simple_brdf_weights.bin";

	std::vector<float>        brdf_weights_vec;
	std::vector<LayerVariant> layers;
	CHECK(load_weights(network_info.file_name.data(), layers, brdf_weights_vec, network_info.header.data()));

	using Component = vk::ComponentTypeKHR;
	using Layout    = vk::CooperativeVectorMatrixLayoutNV;
	// if (layers.size() != expected_layer_count) {
	// 	std::printf("Error loading weights : wrong number of layers\n");
	// 	std::exit(1);
	// }
	networks[u32(BrdfFunctionType::eWeightsInBuffer)].Init(layers);
	networks[u32(BrdfFunctionType::eWeightsInBufferF16)].Init(layers);
	networks[u32(BrdfFunctionType::eCoopVec)].Init(layers);

	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBuffer)].UpdateOffsetsAndSize(device, Layout::eRowMajor, Component::eFloat32, Component::eFloat32));
	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eWeightsInBufferF16)].UpdateOffsetsAndSize(device, Layout::eRowMajor, Component::eFloat16, Component::eFloat16));
	CHECK_VULKAN_RESULT(networks[u32(BrdfFunctionType::eCoopVec)].UpdateOffsetsAndSize(device, Layout::eInferencingOptimal, Component::eFloat16, Component::eFloat16));

	vk::DeviceSize const kMinSize = 256 * 1024;

	std::size_t total_size_bytes = std::max(
		kMinSize,
		vertices_size_aligned + indices_size_aligned
			+ networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize()
			+ networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize()
			+ networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize());

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

	this->vertices_offset = 0;
	this->indices_offset  = this->vertices_offset + vertices_size_aligned;

	// this->linear_weights_offset     = this->indices_offset + AlignUpPowerOfTwo(indices_size_bytes, CoopVecUtils::GetMatrixAlignment());
	// this->linear_weights_offset_f16 = this->linear_weights_offset + AlignUpPowerOfTwo(networks[u32(BrdfFunctionType::eScalarBuffer)].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	// this->optimal_weights_offset    = this->linear_weights_offset_f16 + AlignUpPowerOfTwo(networks[u32(BrdfFunctionType::eScalarBufferF16)].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());

	auto offset = this->indices_offset + indices_size_bytes;
	for (auto i = 1u; i < std::size(networks); ++i) {
		offset             = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());
		weights_offsets[i] = offset;
		offset += AlignUpPowerOfTwo(networks[i].GetParametersSize(), CoopVecUtils::GetMatrixAlignment());
	}

	auto p_staging  = static_cast<std::byte*>(staging_buffer.GetMappedData());
	auto p_vertices = reinterpret_cast<UVSphere::Vertex*>(p_staging + this->vertices_offset);
	auto p_indices  = reinterpret_cast<UVSphere::IndexType*>(p_staging + this->indices_offset);
	sphere.WriteVertices(p_vertices);
	sphere.WriteIndices(p_indices);

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

	write_network.template operator()<float, float>(networks[u32(BrdfFunctionType::eWeightsInBuffer)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)], Layout::eRowMajor, Component::eFloat32, Component::eFloat32);
	write_network.template operator()<float, numeric::float16_t>(networks[u32(BrdfFunctionType::eWeightsInBufferF16)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)], Layout::eRowMajor, Component::eFloat32, Component::eFloat16);
	write_network.template operator()<float, numeric::float16_t>(networks[u32(BrdfFunctionType::eCoopVec)], brdf_weights_src.data(), p_staging + weights_offsets[u32(BrdfFunctionType::eCoopVec)], Layout::eInferencingOptimal, Component::eFloat32, Component::eFloat16);

	if (verbose) {
		// networks[u32(BrdfFunctionType::eScalarBuffer)].Print();
		// networks[u32(BrdfFunctionType::eCoopVec)].Print();
		// PrintLayerBiases<float16_t>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayer<Linear>(networks[u32(BrdfFunctionType::eCoopVec)].GetLayers().size() - 1), p_staging + this->optimal_weights_offset);
		// networks[u32(BrdfFunctionType::eScalarBuffer)].PrintLayerBiases(networks[u32(BrdfFunctionType::eScalarBuffer)].GetLayers().size() - 1, Component::eFloat32, p_staging + this->linear_weights_offset);
	}

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
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBuffer)],
			.size      = networks[u32(BrdfFunctionType::eWeightsInBuffer)].GetParametersSize(),
		},
		// Linear f16
		{
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eWeightsInBufferF16)],
			.size      = networks[u32(BrdfFunctionType::eWeightsInBufferF16)].GetParametersSize(),
		},
		// Optimal
		{
			.srcOffset = weights_offsets[u32(BrdfFunctionType::eCoopVec)],
			.dstOffset = weights_offsets[u32(BrdfFunctionType::eCoopVec)],
			.size      = networks[u32(BrdfFunctionType::eCoopVec)].GetParametersSize(),
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

auto BRDFSample::DrawWindow() -> u64 {
	if (function_id)
		return DrawWindow(pipelines_header[u32(*function_id)]);
	else
		return DrawWindow(pipelines[u32(function_type)]);
};

auto BRDFSample::DrawWindow(vk::Pipeline pipeline) -> u64 {
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

void BRDFSample::RecordCommands(vk::Pipeline pipeline) {
	int x, y, width, height;
	window.GetRect(x, y, width, height);

	auto depth_extent = depth_image.GetExtent();
	if (static_cast<u32>(width) > depth_extent.width || static_cast<u32>(height) > depth_extent.height) {
		depth_image.Recreate({static_cast<u32>(width), static_cast<u32>(height), 1});
	}

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
			.clearValue  = {{{{0.5f, 0.5f, 0.5f, 0.0f}}}},
			// .clearValue = {{{{1.f, 1.f, 1.f, 1.0f}}}},
			// .clearValue  = {{{{0.f, 0.f, 0.f, 1.0f}}}},
		}}},
		.depthAttachment  = {
			 .imageView   = depth_image.GetView(),
			 .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
			 .loadOp      = vk::AttachmentLoadOp::eClear,
			 .storeOp     = vk::AttachmentStoreOp::eDontCare,
			 .clearValue  = {{{{1.0f, 0}}}},
        },
	});
	// auto pipeline = pipelines[u32(function_type)];
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

	camera.getForward() *= -1.0;
	camera.updateProjectionViewInverse();
	camera.getForward() *= -1.0;
	BRDFConstants constants{
		.view_proj = camera.getProjViewInv(),
		.material  = {
			 .base_color = float4{0.1, 0.6, 0.8, 1.0} * 0.2,
			 .metallic   = 0.0f,
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
	};

	// PrintMat4(camera.getView());
	// std::printf("\n");

	// Update weight offsets in push constants
	auto network = networks[u32(function_type)];
	for (int i = 0; i < network.GetLayers().size(); ++i) {
		auto layer                   = network.GetLayer<Linear>(i);
		auto offset_base             = weights_offsets[u32(function_type)];
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

void BRDFSample::SaveSwapchainImageToFile(std::string_view filename) {
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
	// pending_image_save = false;
}

void BRDFSample::Run() {
	do {
		WindowManager::WaitEvents();
		if (window.GetShouldClose()) break;
		int x, y, width, height;
		window.GetRect(x, y, width, height);
		if (width <= 0 || height <= 0) continue;
		u64       elapsed_ns = DrawWindow();
		verbose&& std::printf("%f ms\n", elapsed_ns / 1000000.0);
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

	// bool is_header = true;
	bool is_header = false;
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
#include "FASTKAN_HeaderNames.def"
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
