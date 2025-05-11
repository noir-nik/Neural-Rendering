export module NeuralGraphics:GenericNetwork;
export import :GenericLayers;
import std;

class GenericNetwork {
public:
	GenericNetwork() = default;
	GenericNetwork(std::initializer_list<LayerVariant> layers);
	auto operator=(std::span<LayerVariant> layers) -> GenericNetwork&;

	auto Init(std::span<LayerVariant> layers) -> bool;

	virtual ~GenericNetwork() {}

	auto GetLayers() -> std::span<LayerVariant> { return layers; }
	auto GetLayers() const -> std::span<LayerVariant const> { return layers; }

	auto GetLayer(u32 index) -> LayerVariant& { return layers[index]; }
	auto GetLayer(u32 index) const -> LayerVariant const& { return layers[index]; }

	template <GenericLayerType T>
	auto GetLayer(u32 index) -> T& { return std::get<T>(layers[index]); }

	template <GenericLayerType T>
	auto GetLayer(u32 index) const -> T const& { return std::get<T>(layers[index]); }

	auto GetParametersCount() const -> u32;

private:
	std::vector<LayerVariant> layers;
};
