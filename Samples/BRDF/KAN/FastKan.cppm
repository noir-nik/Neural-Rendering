module;
#include <cassert> // assert

export module FastKan;

import NeuralGraphics;
import std;

export struct KANBuffer {
	// u32 shape;
	u64 offset_; // in elements
	u64 size_;   // in elements

	auto offset() const -> u64 { return offset_; }
	auto size() const -> u64 { return size_; }

	auto span(std::byte const* p) const -> std::span<float const> { return {reinterpret_cast<float const*>(p + offset()), size()}; }
	auto span(float const* p) const -> std::span<float const> { return {reinterpret_cast<float const*>(p + offset()), size()}; }
};

// using KANBuffer = std::span<float const>;

export class FastKanLayer {
	// rbf_grid
	// rbf_denom_inv
	// spline_weight
	// base_weight
	// base_bias

	// KANBuffer buffers[5];
	std::array<KANBuffer, 5> buffers;

public:
	auto size() const -> std::size_t;
	auto size_bytes() const -> std::size_t { return size() * sizeof(float); }

	auto get_buffers() const -> std::span<KANBuffer const> { return buffers; }
	auto get_buffer(u32 index) const -> KANBuffer const& { return buffers[index]; }
	auto get_rbf_grid() const -> KANBuffer const& { return buffers[0]; }
	auto get_rbf_denom_inv() const -> KANBuffer const& { return buffers[1]; }
	auto get_spline_weight() const -> KANBuffer const& { return buffers[2]; }
	auto get_base_weight() const -> KANBuffer const& { return buffers[3]; }
	auto get_base_bias() const -> KANBuffer const& { return buffers[4]; }

	auto get_buffer(u32 index) -> KANBuffer& { return buffers[index]; }
	auto get_rbf_grid() -> KANBuffer& { return buffers[0]; }
	auto get_rbf_denom_inv() -> KANBuffer& { return buffers[1]; }
	auto get_spline_weight() -> KANBuffer& { return buffers[2]; }
	auto get_base_weight() -> KANBuffer& { return buffers[3]; }
	auto get_base_bias() -> KANBuffer& { return buffers[4]; }

	auto get_buffer_name(u32 index) -> std::string_view;

	void repr() const;

	void repr_buffer(std::span<float const> buffer) const;
};


export struct FastKan {
	std::vector<FastKanLayer> layers_;
	std::vector<float>        buffer_;

	auto layers() const -> std::span<FastKanLayer const> { return layers_; }
	auto layers() -> std::span<FastKanLayer> { return layers_; }
	auto buffer() const -> std::span<float const> { return buffer_; }
	auto buffer() -> std::span<float> { return buffer_; }

	void repr() const {
		std::printf("FastKan {\n");
		for (u32 i = 0; i < layers().size(); ++i) {
			std::printf("  layer[%u] = {\n", i);
			layers()[i].repr();
			std::printf("  }\n");
		}
		std::printf("}\n");
	}

	auto size() const -> std::size_t;
	auto size_bytes() const -> std::size_t { return size() * sizeof(float); }
};
