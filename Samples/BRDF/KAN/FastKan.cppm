module;
#include <cassert> // assert

export module FastKan;

import NeuralGraphics;
import std;

export struct KANBuffer {
	// u32 shape;
	u64 offset; // in elements
	u64 size;   // in elements

	auto span(std::byte const* p) const -> std::span<float const> { return {reinterpret_cast<float const*>(p + offset), size}; }
	auto span(float const* p) const -> std::span<float const> { return {reinterpret_cast<float const*>(p + offset), size}; }
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

	auto get_buffer_name(u32 index) -> std::string_view {
		switch (index) {
		case 0:  return "rbf_grid";
		case 1:  return "rbf_denom_inv";
		case 2:  return "spline_weight";
		case 3:  return "base_weight";
		case 4:  return "base_bias";
		default: {
			assert(false && "Invalid index");
			return "";
		};
		}
	}

	void repr() const {
#define _rpr(name) \
	std::printf(#name ": %6zu, size: %6zu\n", get_##name().offset, get_##name().size);

		_rpr(rbf_grid);
		_rpr(rbf_denom_inv);
		_rpr(spline_weight);
		_rpr(base_weight);
		_rpr(base_bias);
#undef _rpr
	}

	void repr_buffer(std::span<float const> buffer) const {
		if (buffer.size() == 0) {
			std::printf("Empty buffer\n");
			return;
		}

		std::printf("FastKanLayer {\n");
		for (u32 i = 0; i < buffers.size(); ++i) {
			std::printf("  buffer[%u] = {\n", i);
			auto span = buffers[i].span(buffer.data());
			std::printf("%f", span[0]);
			if (span.size() > 1)
				for (u32 j = 1; j < span.size(); ++j) {
					std::printf(", %f", span[j]);
				}
			std::printf("  }\n");
		}
		std::printf("}\n");
	}
};

export struct FastKan {
	std::vector<FastKanLayer> layers;
	std::vector<float>        buffer;

	void repr() const {
		std::printf("FastKan {\n");
		for (u32 i = 0; i < layers.size(); ++i) {
			std::printf("  layer[%u] = {\n", i);
			layers[i].repr();
			std::printf("  }\n");
		}
		std::printf("}\n");
	}
};
