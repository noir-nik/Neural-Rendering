module;
#include <cassert> // assert
module FastKan;

import NeuralGraphics;
import std;

auto FastKanLayer::size() const -> std::size_t {
	return std::ranges::fold_left(get_buffers(), std::size_t(0), [](auto acc, auto&& x) -> std::size_t {
		return acc + x.size();
	});
}

void FastKanLayer::repr() const {
#define _rpr(name) \
	std::printf("    %-16s offset: %-6zu size: %-6zu\n", #name, get_##name().offset(), get_##name().size());

	_rpr(rbf_grid);
	_rpr(rbf_denom_inv);
	_rpr(spline_weight);
	_rpr(base_weight);
	_rpr(base_bias);
#undef _rpr
}

void FastKanLayer::repr_buffer(std::span<float const> buffer) const {
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

auto FastKanLayer::get_buffer_name(u32 index) -> std::string_view {
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

auto FastKan::size() const -> std::size_t {
	return std::ranges::fold_left(layers_, std::size_t(0), [](auto acc, auto&& x) -> std::size_t {
		return acc + x.size();
	});
}

void FastKan::repr() const {
	std::printf("FastKan {\n");
	for (u32 i = 0; i < layers().size(); ++i) {
		std::printf("  layer[%u] = {\n", i);
		layers()[i].repr();
		std::printf("  }\n");
	}
	std::printf("}\n");
}
