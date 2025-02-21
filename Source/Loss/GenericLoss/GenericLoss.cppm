export module NeuralGraphics:GenericLoss;
import :Core;
import std;

export namespace ng::Loss {

// template<typename Backend>
class MeanSquaredError {
public:
	template<typename T>
	auto operator()(std::span<T const> outputs, std::span<T const> targets) const -> T {
		T loss = T{0};
		for (u32 i = 0; i < outputs.size(); ++i) {
			loss += (outputs[i] - targets[i]) * (outputs[i] - targets[i]);
		}
		return loss / outputs.size();
	}

	template<typename T>
	void Backward(std::span<T const> outputs, std::span<T const> targets, T* gradOutputs) const {
		for (u32 i = 0; i < outputs.size(); ++i) {
			gradOutputs[i] = (outputs[i] - targets[i]) * T{2} / outputs.size();
		}
	}
};
class CrossEntropy {};

} // namespace ng
