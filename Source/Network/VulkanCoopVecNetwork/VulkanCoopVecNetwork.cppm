export module NeuralGraphics:VulkanCoopVecNetwork;
import :GenericNetwork;
import :Core;
import :Utils;
import std;
import vulkan_hpp;

export namespace ng {

template <typename T>
concept VulkanCoopVecNetworkType = IsAnyV<T, float>;

template <VulkanCoopVecNetworkType T>
constexpr auto GetVulkanComponentType() -> vk::ComponentTypeKHR {
	if constexpr (std::same_as<T, float>) {
		return vk::ComponentTypeKHR::eFloat32;
	}
}

template <typename T>
constexpr auto AlignTo(T const value, T const alignment) -> T {
	return ((value + alignment - T(1)) / alignment) * alignment;
}

template <VulkanCoopVecNetworkType WeightsT = float, VulkanCoopVecNetworkType BiasesT = float>
class VulkanCoopVecNetwork : public GenericNetwork {
public:
	using WeightsType = WeightsT;
	using BiasesType  = BiasesT;

	VulkanCoopVecNetwork(std::initializer_list<LayerVariant> layers) : GenericNetwork(layers) {};
	auto GetParametersSize() const -> std::size_t { return parameters_size; }

	// auto GetLayerWeights(Linear const& layer) -> std::span<WeightsType> {
	// 	return {
	// 		reinterpret_cast<WeightsType*>(&parameters[layer.GetWeightsOffset()]),
	// 		layer.GetWeightsCount() * sizeof(WeightsType),
	// 	};
	// }
	// auto GetLayerBiases(Linear const& layer) -> std::span<BiasesType> {
	// 	return {
	// 		reinterpret_cast<BiasesType*>(&parameters[layer.GetBiasesOffset()]),
	// 		layer.GetBiasesCount() * sizeof(BiasesType),
	// 	};
	// }

	[[nodiscard]] auto Init(vk::Device device) -> vk::Result {
		this->device = device;
		if (vk::Result result = CalculateOffsets(); result != vk::Result::eSuccess) return result;

		return vk::Result::eSuccess;
	}

private:
	[[nodiscard]] auto CalculateOffsets() -> vk::Result;

	vk::Device                          device;
	vk::CooperativeVectorMatrixLayoutNV layout          = vk::CooperativeVectorMatrixLayoutNV::eRowMajor;
	std::size_t                         parameters_size = 0;
};
} // namespace ng

namespace ng {
template <VulkanCoopVecNetworkType WeightsType, VulkanCoopVecNetworkType BiasesType>
auto VulkanCoopVecNetwork<WeightsType, BiasesType>::CalculateOffsets() -> vk::Result {
	std::size_t  offset                = 0;
	u32          current_layer_outputs = std::get<Linear>(GetLayer(0)).GetOutputsCount();
	CoopVecUtils coop_vec_utils(device);
	for (LayerVariant& layer : GetLayers()) {
		std::visit(
			Visitor{
				[&offset, &current_layer_outputs, this, &coop_vec_utils](Linear& layer) -> vk::Result {
					auto [result, size] = coop_vec_utils.CalculateByteSize(layer.GetOutputsCount(), layer.GetInputsCount(), layout, GetVulkanComponentType<WeightsType>());
					if (result != vk::Result::eSuccess) return result;
					offset = AlignTo(offset, coop_vec_utils.GetMatrixAlignment());
					layer.SetWeightsOffset(offset);
					offset += size;
					offset = AlignTo(offset, coop_vec_utils.GetVectorAlignment());
					layer.SetBiasesOffset(offset);
					offset += layer.GetOutputsCount() * GetVulkanComponentSize(GetVulkanComponentType<BiasesType>());

					current_layer_outputs = layer.GetOutputsCount();
					return vk::Result::eSuccess;
				},
				[&current_layer_outputs, this](auto& layer) -> vk::Result {
					layer.SetInputsCount(current_layer_outputs);
					return vk::Result::eSuccess;
				}},
			layer);
	}
	parameters_size = offset;
	return vk::Result::eSuccess;
}

} // namespace ng

namespace {
void VulkanCoopVecNetworkTest() {
	ng::VulkanCoopVecNetwork<float, float> network({
		ng::Linear(3, 16),
		ng::Sin(),
		ng::Linear(16, 16),
		ng::Sin(),
		ng::Linear(16, 16),
		ng::Sin(),
		ng::Linear(16, 1),
	});

	// network.GetLayerWeights(network.GetLayer<ng::Linear>(0));
}
} // namespace
