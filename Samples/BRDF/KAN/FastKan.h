#pragma once

import NeuralGraphics;
import std;

#ifndef EXPORT
#define EXPORT
#endif

EXPORT struct KANBuffer {
	// u32 shape;
	u64 offset_; // in elements
	u64 size_;   // in elements

	static constexpr u32 kMaxShapeSize = 4;
	std::array<u32, 4>   shape_;
	u32                  shape_size_ = 0;

	auto offset() const -> u64 { return offset_; }
	auto size() const -> u64 { return size_; }
	auto shape() const -> std::span<u32 const> { return {shape_.data(), shape_size_}; }

	template <typename Rg>
	auto set_shape(Rg&& r) -> decltype(*this) {
		for (auto&& elem : r)
			shape_[shape_size_++] = elem;
		return *this;
	}

	auto span(std::byte const* p) const -> std::span<float const> { return {reinterpret_cast<float const*>(p) + offset(), size()}; }
	auto span(float const* p) const -> std::span<float const> { return {p + offset(), size()}; }

	auto size_bytes() const -> std::size_t { return size() * sizeof(float); }
};

// using KANBuffer = std::span<float const>;


EXPORT class KanLayerBaseBase {
public:
	// auto get_buffer_name(u32 index) const -> std::string_view;
};

EXPORT template <typename T, std::size_t N>
class KanLayerBase : public KanLayerBaseBase {
	// KANBuffer buffers[5];
	std::array<T, N> buffers_;

public:
	using value_type = T;
	auto get_buffers() const -> std::span<T const> { return buffers_; }
	auto get_buffer(u32 index) const -> T const& { return buffers_[index]; }

	auto get_buffers() -> std::span<T> { return buffers_; }
	auto get_buffer(u32 index) -> T& { return buffers_[index]; }
};

EXPORT template <typename T>
class FastKanLayerBase : public KanLayerBase<T, 5> {
	using Base = KanLayerBase<T, 5>;

public:
	auto get_rbf_grid() const -> T const& { return Base::get_buffer(0); }
	auto get_rbf_denom_inv() const -> T const& { return Base::get_buffer(1); }
	auto get_spline_weight() const -> T const& { return Base::get_buffer(2); }
	auto get_base_weight() const -> T const& { return Base::get_buffer(3); }
	auto get_base_bias() const -> T const& { return Base::get_buffer(4); }

	auto get_rbf_grid() -> T& { return Base::get_buffer(0); }
	auto get_rbf_denom_inv() -> T& { return Base::get_buffer(1); }
	auto get_spline_weight() -> T& { return Base::get_buffer(2); }
	auto get_base_weight() -> T& { return Base::get_buffer(3); }
	auto get_base_bias() -> T& { return Base::get_buffer(4); }
};

EXPORT class FastKanLayer : public FastKanLayerBase<KANBuffer> {
public:
	auto get_buffer_name(u32 index) const -> std::string_view;
	auto size() const -> std::size_t;
	auto size_bytes() const -> std::size_t { return size() * sizeof(float); }
	void repr() const;

	void repr_buffer(std::span<float const> buffer) const;
};

EXPORT struct FastKan {
	std::vector<FastKanLayer> layers_;
	std::vector<float>        buffer_;

	auto layers() const -> std::span<FastKanLayer const> { return layers_; }
	auto layers() -> std::span<FastKanLayer> { return layers_; }
	auto buffer() const -> std::span<float const> { return buffer_; }
	auto buffer() -> std::span<float> { return buffer_; }

	void repr() const;

	auto size() const -> std::size_t;
	auto size_bytes() const -> std::size_t { return size() * sizeof(float); }
};
