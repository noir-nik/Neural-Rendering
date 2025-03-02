export module NeuralGraphics:HostNetwork;
import :GenericNetwork;
import :HostLayers;
import :Core;
import :Utils;
import std;

export namespace ng {

template <typename T>
class HostNetwork : public GenericNetwork {
public:
	using ValueType = T;
	HostNetwork(std::initializer_list<LayerVariant> layers) : GenericNetwork(layers) {};
	auto GetParameters() const -> std::span<ValueType const> { return parameters; }

	auto GetLayerWeights(Linear const& layer) -> ValueType* { return &parameters[layer.GetWeightsOffset()]; }
	auto GetLayerBiases(Linear const& layer) -> ValueType* { return &parameters[layer.GetBiasesOffset()]; }

	auto GetParametersSize() const -> u32;
	auto GetActivationsCount() const -> u32;
	auto GetLayerOutputsCount() const -> u32;
	auto GetScratchBufferCount() const -> u32 { return GetLayerOutputsCount() + GetActivationsCount(); }

	auto Forward(std::span<ValueType const> inputs, ValueType* scratchBuffer) -> std::span<ValueType>;
	auto Backward(std::span<ValueType const> lossGradients, ValueType* gradientBuffer) -> std::span<ValueType>;

private:
	std::vector<ValueType> parameters;
};
} // namespace ng

export namespace ng {
template <typename T>
auto HostNetwork<T>::Forward(std::span<ValueType const> inputs, ValueType* scratchBuffer) -> std::span<ValueType> {}

} // namespace ng

namespace {
void HostNetworkTest() {
	ng::HostNetwork<float> model({
		ng::Linear(28 * 28, 100),
		ng::Relu(),
		ng::Linear(100, 10),
	});
}
} // namespace
