export module NeuralGraphics:GenericNetwork;
export import :GenericLayers;

import std;
namespace ng {
export class GenericNetwork {
public:
	GenericNetwork(std::initializer_list<LayerVariant> layers);
	virtual ~GenericNetwork() {}

	auto GetLayers() -> std::span<LayerVariant const> { return layers; }
	auto GetLayer(u32 index) -> LayerVariant const& { return layers[index]; }
	auto GetParametersSize() const -> u32;
	auto GetActivationsSize() const -> u32;
	auto GetLayerOutputsSize() const -> u32;
	auto GetScratchBufferSize() const -> u32 { return GetLayerOutputsSize() + GetActivationsSize(); }

private:
	std::vector<LayerVariant> layers;
	void CalculateOffsetsAndSizes();
};

} // namespace ng
