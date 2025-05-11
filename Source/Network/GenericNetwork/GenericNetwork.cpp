module;

#include <cassert>

module NeuralGraphics;
import :GenericNetwork;
import :Utils;
import std;

void ValidateLayers(std::span<LayerVariant const> layers) {
	assert(!layers.empty() && "Network must have at least one layer.");
	assert(layers.size() > 1 && "Network must have at least two layers.");
	assert(std::holds_alternative<Linear>(layers[0]) && "First layer must be a linear layer.");
	u32 currentSize = std::get<Linear>(layers[0]).GetOutputsCount();
	for (auto& layer : layers | std::views::drop(1)) {
		if (std::holds_alternative<Linear>(layer)) {
			auto linear = std::get<Linear>(layer);
			assert(linear.GetInputsCount() == currentSize && "Layer input size must match previous layer output size.");
			currentSize = linear.GetOutputsCount();
		}
	}
}

GenericNetwork::GenericNetwork(std::initializer_list<LayerVariant> layers) : layers(layers) {
	ValidateLayers(layers);
}

auto GenericNetwork::operator=(std::span<LayerVariant> layers) -> GenericNetwork& {
	bool result = Init(layers);
	(void)result;
	return *this;
};

auto GenericNetwork::Init(std::span<LayerVariant> layers) -> bool {
	this->layers.assign(layers.begin(), layers.end());
	ValidateLayers(layers);
	return true;
}

auto GenericNetwork::GetParametersCount() const -> u32 {
	u32 size = 0;
	for (auto const& layer : GetLayers()) {
		if (std::holds_alternative<Linear>(layer)) {
			size += std::get<Linear>(layer).GetParametersCount();
		}
	}
	return size;
}
