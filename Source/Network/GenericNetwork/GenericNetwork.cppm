export module NeuralGraphics:GenericNetwork;
export import :GenericLayers;
import std;

export namespace ng {
class GenericNetwork {
public:
	GenericNetwork(std::initializer_list<LayerVariant> layers);
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

} // namespace ng
