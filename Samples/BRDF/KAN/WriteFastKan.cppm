export module FastKANCoopVec:Write;
import :FastKAN;

import std;
import vulkan;

using ComponentTy = vk::ComponentTypeKHR;
using LayoutTy    = vk::CooperativeVectorMatrixLayoutNV;

export using FastKanOffsets = std::vector<FastKanLayerBase<u64>>;

struct FastKanLayerOffsets {
	FastKanLayerBase<u64> offsets;
	std::size_t           total_size = 0;
};

export auto write_fast_kan_layer(
	vk::Device             device,
	FastKanLayer const&    layer,
	std::span<float const> src_buffer,
	std::byte*             dst_parameters,
	LayoutTy               dst_layout,
	ComponentTy            src_component_type,
	ComponentTy            dst_matrix_type)
	-> FastKanLayerOffsets;

export auto write_fast_kan(
	vk::Device     device,
	FastKan const& kan,
	std::byte*     dst_parameters,
	LayoutTy       dst_layout,
	ComponentTy    src_component_type,
	ComponentTy    dst_matrix_type)
	-> FastKanOffsets;
