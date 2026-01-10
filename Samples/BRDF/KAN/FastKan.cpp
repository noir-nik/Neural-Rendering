module;
#include <cassert> // assert
module FastKan;

import NeuralGraphics;
import std;

auto FastKanLayer::get_buffer_name(u32 index) const -> std::string_view {
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

auto FastKanLayer::size() const -> std::size_t {
	return std::ranges::fold_left(get_buffers(), std::size_t(0), [](auto acc, auto&& x) -> std::size_t {
		return acc + x.size();
	});
}

void FastKanLayer::repr() const {

	for (u32 i = 0; i < get_buffers().size(); ++i) {
		auto const& buf = get_buffers()[i];
		std::printf("    %-16s offset: %-6zu size: %-6zu shape: [", get_buffer_name(i).data(), buf.offset(), buf.size());
		for (u32 j = 0; j < buf.shape_size_; ++j) {
			if (j > 0)
				std::printf(", ");
			std::printf("%u", buf.shape()[j]);
		}
		std::printf("]\n");
	}
}

void FastKanLayer::repr_buffer(std::span<float const> buffer) const {
	if (buffer.size() == 0) {
		std::printf("Empty buffer\n");
		return;
	}

	std::printf("FastKanLayer {\n");
	for (u32 i = 0; i < get_buffers().size(); ++i) {
		std::printf("  buffer[%u] = {\n", i);
		auto span = get_buffers()[i].span(buffer.data());
		std::printf("%f", span[0]);
		if (span.size() > 1)
			for (u32 j = 1; j < span.size(); ++j) {
				std::printf(", %f", span[j]);
			}
		std::printf("  }\n");
	}
	std::printf("}\n");
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
