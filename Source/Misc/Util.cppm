export module NeuralGraphics:Util;
import std;

export namespace ng {

template <class... Ts>
struct Visitor : Ts... {
	using Ts::operator()...;
};

template <typename T, typename... Ts>
using IsAny = std::disjunction<std::is_same<T, Ts>...>;

template <typename T, typename... Ts>
inline constexpr bool IsAnyV = IsAny<T, Ts...>::value;

} // namespace ng
