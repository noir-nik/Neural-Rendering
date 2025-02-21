export module NeuralGraphics:Network;
export import :Network.Layer;

import std;

export
class Network {
private:
	std::vector<NetworkLayerVariant> layers;
public:
	Network(std::span<NetworkLayerVariant const> layers);

	auto GetLayers() -> std::span<NetworkLayerVariant const> { return layers; }
};
