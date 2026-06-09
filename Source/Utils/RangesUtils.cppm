export module NeuralGraphics:RangesUtils;

import std;

export namespace Utils {
constexpr inline auto indices = [] [[nodiscard]] (std::size_t size) static {
	return std::views::iota(decltype(size){}, size);
};
} // namespace Utils
