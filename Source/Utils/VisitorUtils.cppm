export module NeuralGraphics:VisitorUtils;
import std;

export namespace ng {

template <class... Ts>
struct Visitor : Ts... {
	using Ts::operator()...;
};

} // namespace ng
