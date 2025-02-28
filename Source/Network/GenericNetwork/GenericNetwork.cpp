module;

#include <cassert>

module NeuralGraphics;
import :GenericNetwork;
import :Utils;
import std;
namespace ng {

void ValidateLayers(std::span<LayerVariant const> layers) {
	assert(!layers.empty() && "Network must have at least one layer.");
	assert(layers.size() > 1 && "Network must have at least two layers.");
	assert(std::holds_alternative<Linear>(layers[0]) && "First layer must be a linear layer.");
	u32 currentSize = std::get<Linear>(layers[0]).GetOutputSize();
	for (auto& layer : layers) {
		if (std::holds_alternative<Linear>(layer)) {
			auto linear = std::get<Linear>(layer);
			assert(linear.GetInputSize() == currentSize && "Layer input size must match previous layer output size.");
			currentSize = linear.GetOutputSize();
		}
	}
}

void GenericNetwork::CalculateOffsetsAndSizes() {
	u32 offset = 0;
	u32 currentLayerOutputSize = std::get<Linear>(layers[0]).GetOutputSize();
	for (LayerVariant& layer : layers) {
		std::visit(Visitor{[&offset, &currentLayerOutputSize](Linear& layer) {
							   layer.SetWeightsOffset(offset);
							   offset += layer.GetWeightsSize();
							   layer.SetBiasesOffset(offset);
							   offset += layer.GetBiasesSize();
							   currentLayerOutputSize = layer.GetOutputSize();
						   },
						   [&currentLayerOutputSize](auto& layer) { layer.SetInputSize(currentLayerOutputSize); }},
				   layer);
	}
}

GenericNetwork::GenericNetwork(std::initializer_list<LayerVariant> layers) : layers(layers) {
	ValidateLayers(layers);
	CalculateOffsetsAndSizes();
}

auto GenericNetwork::GetParametersSize() const -> u32 {
	u32 size = 0;
	for (auto& layer : layers) {
		if (std::holds_alternative<Linear>(layer)) {
			size += std::get<Linear>(layer).GetParametersSize();
		}
	}
	return size;
}

auto GenericNetwork::GetActivationsSize() const -> u32 {
	u32 size = std::get<Linear>(layers[0]).GetInputSize();
	for (auto& layer : layers) {
		if (std::holds_alternative<Linear>(layer)) {
			size += std::get<Linear>(layer).GetOutputSize();
		}
	}
	return size;
}
auto GenericNetwork::GetLayerOutputsSize() const -> u32 {
	u32 size = 0;
	for (auto& layer : layers) {
		if (std::holds_alternative<Linear>(layer)) {
			size += std::get<Linear>(layer).GetOutputSize();
		}
	}
	return size;
}

} // namespace ng
