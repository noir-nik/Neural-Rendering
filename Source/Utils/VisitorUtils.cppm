export module NeuralGraphics:VisitorUtils;
import std;

export namespace Utils {

template <class... Ts>
struct Visitor : Ts... {
	using Ts::operator()...;
};

} // namespace Utils
