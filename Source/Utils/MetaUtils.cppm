export module NeuralGraphics:MetaUtils;
import std;

export namespace Utils {

template <typename T, typename... Ts>
using IsAny = std::disjunction<std::is_same<T, Ts>...>;

template <typename T, typename... Ts>
inline constexpr bool IsAnyV = IsAny<T, Ts...>::value;

} // namespace Utils
