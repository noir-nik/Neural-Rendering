export module NeuralGraphics:HostOptimizers;
import :GenericOptimizer;
import std;

export namespace ng::HostOptimizers {

template <typename T>
class HostOptimizer : public GenericOptimizer {
public:
	HostOptimizer(std::span<T> parameters, std::span<T> gradients) : parameters(parameters), gradients(gradients) {};
	HostOptimizer() = delete;

protected:
	std::span<T> parameters;
	std::span<T> gradients;
};

template <typename T>
class SGD : public HostOptimizer<T> {
public:
	SGD(std::span<T> parameters) : HostOptimizer<T>(parameters) {}

private:
	void Step() override;
};

} // namespace ng::HostOptimizers

namespace ng::HostOptimizers {
template <typename T>
void SGD<T>::Step() {
	for (u32 i = 0; i < this->parameters.size(); ++i) {
		this->parameters[i] -= this->gradients[i];
	}
}
} // namespace ng::HostOptimizers
