module;
#include "CheckResult.h"
module FastKANCoopVec;
import :Write;

import std;
import vulkan;

import SamplesCommon;
using numeric::float16_t;

using ComponentTy = vk::ComponentTypeKHR;
using LayoutTy    = vk::CooperativeVectorMatrixLayoutNV;

static constexpr auto kVectorAlignment = CoopVecUtils::GetVectorAlignment();

struct MatrixInfo {
	vk::Device  device;
	void const* src;
	std::size_t src_size;
	std::byte*  dst;
	LayoutTy    dst_layout;
	ComponentTy src_component_type;
	ComponentTy dst_matrix_type;
	u32         rows;
	u32         cols;
};

auto write_matrix_kan(MatrixInfo const& info) -> std::size_t {
	std::size_t required_size;

	auto const& [device, src, src_size, dst, dst_layout, src_component_type, dst_matrix_type, rows, cols] = info;

	vk::ConvertCooperativeVectorMatrixInfoNV create_info{
		.srcSize          = info.src_size,
		.srcData          = {.hostAddress = info.src},
		.pDstSize         = &required_size,
		.dstData          = {.hostAddress = info.dst},
		.srcComponentType = info.src_component_type,
		.dstComponentType = info.dst_matrix_type,
		.numRows          = info.rows,
		.numColumns       = info.cols,
		.srcLayout        = vk::CooperativeVectorMatrixLayoutNV::eRowMajor,
		.srcStride        = info.cols * Utils::GetVulkanComponentSize(info.src_component_type),
		.dstLayout        = info.dst_layout,
		.dstStride        = info.cols * Utils::GetVulkanComponentSize(info.dst_matrix_type),
	};

	CHECK_VULKAN_RESULT(info.device.convertCooperativeVectorMatrixNV(&create_info));

	std::printf("Src size: %zu, result size: %zu\n", info.src_size, required_size);

	return required_size;
};

auto write_fast_kan_layer(
	vk::Device             device,
	FastKanLayer const&    layer,
	std::span<float const> src_buffer,
	std::byte*             dst_parameters,
	LayoutTy               dst_layout,
	ComponentTy            src_component_type,
	ComponentTy            dst_matrix_type)
	-> FastKanLayerOffsets {

	FastKanLayerOffsets result;

	auto& offset = result.total_size;
	// auto offset = std::size_t{0};

	auto& buffer_offsets = result.offsets;

	auto wrt = [&](KANBuffer const& buffer) {
		auto sspan    = buffer.span(src_buffer.data());
		auto dstns    = std::distance(&layer.get_buffers()[0], &buffer);
		auto dst_size = std::size_t{buffer.size() * sizeof(float16_t)};
		std::printf("Writing buffer %s %td/%zu, offset: %zu, size: %zu, dst_size: %zu\n", layer.get_buffer_name(dstns).data(), dstns, layer.get_buffers().size(), offset, sspan.size_bytes(), dst_size);
		// write_fast_kan_buffer(buffer, dst_parameters + offset);
		{
			if (buffer.offset() + buffer.size() > src_buffer.size()) {
				std::printf("Buffer size too small 1\n");
				std::printf("Buffer offset: %zu, buffer size: %zu, src_buffer size: %zu\n", buffer.offset(), buffer.size(), src_buffer.size());
				std::exit(1);
			}
			std::byte* dst = dst_parameters + offset;
			// std::memcpy(dst, sspan.data(), sspan.size_bytes());
			// float -> float16
			for (u32 i = 0; i < buffer.size(); ++i) {
				*(reinterpret_cast<float16_t*>(dst) + i) = static_cast<float16_t>(sspan[i]);
			}
		}
		return AlignUpPowerOfTwo(offset + dst_size, kVectorAlignment);
	};

	// std::printf("Writing rbf_grid, offset: %zu\n", offset);
	buffer_offsets.rbf_grid() = offset;
	offset                    = wrt(layer.rbf_grid());
	// std::printf("Writing rbf_denom_inv, offset: %zu\n", offset);
	buffer_offsets.rbf_denom_inv() = offset;
	offset                         = wrt(layer.rbf_denom_inv());
	// std::printf("Writing base_bias, offset: %zu\n", offset);
	buffer_offsets.base_bias() = offset;
	offset                     = wrt(layer.base_bias());
	offset                     = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());

	// ret.get_spline_weight() = wrt(layer.get_spline_weight());
	// ret.get_base_weight()   = wrt(layer.get_base_weight());
	// for (u32 i = 0; auto const& buffer : {layer.get_rbf_grid(), layer.get_rbf_denom_inv(), layer.get_base_bias()})

	// Write 2 matrices: [spline_weight base_weight]

	// auto const base_inputs = layer.get_base_weight().shape()[1];
	// auto const base_ouputs = layer.get_base_weight().shape()[0];
	// auto const base_cols   = base_inputs;
	// auto const base_rows   = base_ouputs;

	std::size_t matrix_size;
	std::printf("Writing spline_weight, offset: %zu\n", offset);
	matrix_size = write_matrix_kan({
		.device             = device,
		.src                = layer.spline_weight().span(src_buffer.data()).data(),
		.src_size           = layer.spline_weight().size_bytes(),
		.dst                = dst_parameters + offset,
		.dst_layout         = dst_layout,
		.src_component_type = src_component_type,
		.dst_matrix_type    = dst_matrix_type,
		.rows               = layer.spline_weight().shape()[0],
		.cols               = layer.spline_weight().shape()[1],
	});

	buffer_offsets.spline_weight() = offset;
	offset                         = AlignUpPowerOfTwo(offset + matrix_size, CoopVecUtils::GetMatrixAlignment());

	std::printf("Writing base_weight, offset: %zu\n", offset);
	matrix_size = write_matrix_kan({
		.device             = device,
		.src                = layer.base_weight().span(src_buffer.data()).data(),
		.src_size           = layer.base_weight().size_bytes(),
		.dst                = dst_parameters + offset,
		.dst_layout         = dst_layout,
		.src_component_type = src_component_type,
		.dst_matrix_type    = dst_matrix_type,
		.rows               = layer.base_weight().shape()[0],
		.cols               = layer.base_weight().shape()[1],
	});

	buffer_offsets.base_weight() = offset;
	offset                       = AlignUpPowerOfTwo(offset + matrix_size, CoopVecUtils::GetMatrixAlignment());

	return result;
}

auto write_fast_kan(
	vk::Device     device,
	FastKan const& kan,
	std::byte*     dst_parameters,
	LayoutTy       dst_layout,
	ComponentTy    src_component_type,
	ComponentTy    dst_matrix_type)
	-> FastKanOffsets {
	auto layers = kan.layers();

	auto base_offset = std::size_t{0};

	FastKanOffsets result;
	result.reserve(layers.size());

	for (auto& layer : layers) {
		std::printf("Writing layer, offset: %zu\n", base_offset);
		auto buffer_offsets = write_fast_kan_layer(device, layer, kan.buffer(), dst_parameters + base_offset, dst_layout, src_component_type, dst_matrix_type);
		for (u32 i = 0; i < 5; ++i) {
			buffer_offsets.offsets.get_buffer(i) += base_offset;
		}

		result.push_back(buffer_offsets.offsets);
		base_offset += buffer_offsets.total_size;
	}

	return result;
}
