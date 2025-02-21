export module NeuralGraphics:HostNetwork;
import :GenericNetwork;
import :HostLayers;
import :Core;
import :Util;
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

	auto Forward(std::span<ValueType const> inputs, ValueType* scratchBuffer) -> std::span<ValueType>;
	auto Backward(std::span<ValueType const> lossGradients, ValueType* gradientBuffer) -> std::span<ValueType>;

private:
	std::vector<ValueType> parameters;
};
} // namespace ng

export namespace ng {
template <typename T>
auto HostNetwork<T>::Forward(std::span<ValueType const> inputs, ValueType* scratchBuffer) -> std::span<ValueType> {
	ValueType* pOutput      = scratchBuffer;
	ValueType* pActivations = pOutput + GetLayerOutputsSize();
	std::memcpy(pActivations, inputs.data(), inputs.size() * sizeof(ValueType));
	for (LayerVariant const& layer : GetLayers()) {
		std::visit(Visitor{[&pOutput, &pActivations, this](Linear const& layer) {
							   HostLinear<T>(layer).Forward(pActivations, pOutput, GetLayerWeights(layer), GetLayerBiases(layer));
							   pActivations += layer.GetInputSize();
						   },
						   [&pOutput, &pActivations](Relu const& layer) {
							   HostRelu<T>(layer).Forward(pOutput, pActivations);
							   pOutput += layer.GetOutputSize();
						   },
						   [&pOutput, &pActivations](Sigmoid const& layer) {
							   HostSigmoid<T>(layer).Forward(pOutput, pActivations);
							   pOutput += layer.GetOutputSize();
						   },
						   [&pOutput, &pActivations](Softmax const& layer) {
							   HostSoftmax<T>(layer).Forward(pOutput, pActivations);
							   pOutput += layer.GetOutputSize();
						   }},
				   layer);
	}
	
	u32 outputSize = std::visit([](auto const& layer) -> u32 { return layer.GetOutputSize(); }, GetLayers().back());
	return std::span<ValueType>(pActivations - outputSize, outputSize);
}

void Dummy() {
	ng::HostNetwork<float> model({
		ng::Linear(28 * 28, 100),
		ng::Relu(),
		ng::Linear(100, 10),
	});

	model.Forward({}, {});
}

} // namespace ng
