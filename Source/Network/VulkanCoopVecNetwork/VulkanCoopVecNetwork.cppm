export module NeuralGraphics:VulkanCoopVecNetwork;
import :GenericNetwork;
import :Core;
import :Utils;
import std;
import vulkan_hpp;

export namespace ng {

template <typename T>
concept VulkanCoopVecNetworkType = IsAnyV<T, float>;

// template <typename  T>
// constexpr auto GetVulkanComponentType() -> vk::ComponentTypeKHR {
// 	if constexpr (std::same_as<T, float>) {
// 		return vk::ComponentTypeKHR::eFloat32;
// 	}
// }
struct float16_t;
template <typename T>
inline constexpr auto GetVulkanComponentType() -> vk::ComponentTypeKHR {
	if constexpr (std::is_same_v<T, float16_t>) {
		return vk::ComponentTypeKHR::eFloat16;
	} else if constexpr (std::is_same_v<T, float>) {
		return vk::ComponentTypeKHR::eFloat32;
	} else if constexpr (std::is_same_v<T, std::int8_t>) {
		return vk::ComponentTypeKHR::eSint8;
	} else if constexpr (std::is_same_v<T, std::int16_t>) {
		return vk::ComponentTypeKHR::eSint16;
	} else if constexpr (std::is_same_v<T, std::int32_t>) {
		return vk::ComponentTypeKHR::eSint32;
	} else if constexpr (std::is_same_v<T, std::int64_t>) {
		return vk::ComponentTypeKHR::eSint64;
	} else if constexpr (std::is_same_v<T, std::uint8_t>) {
		return vk::ComponentTypeKHR::eUint8;
	} else if constexpr (std::is_same_v<T, std::uint16_t>) {
		return vk::ComponentTypeKHR::eUint16;
	} else if constexpr (std::is_same_v<T, std::uint32_t>) {
		return vk::ComponentTypeKHR::eUint32;
	} else if constexpr (std::is_same_v<T, std::uint64_t>) {
		return vk::ComponentTypeKHR::eUint64;
	}

	static_assert(false, "Unsupported type.");
}

template <typename T>
constexpr auto AlignTo(T const value, T const alignment) -> T {
	return ((value + alignment - T(1)) / alignment) * alignment;
}

class VulkanCoopVecNetwork : public GenericNetwork {
public:

	VulkanCoopVecNetwork(std::initializer_list<LayerVariant> layers) : GenericNetwork(layers) {};
	auto GetParametersSize() const -> std::size_t { return parameters_size; }

	[[nodiscard]] auto UpdateOffsetsAndSize(vk::Device                          device,
											vk::CooperativeVectorMatrixLayoutNV layout,
											vk::ComponentTypeKHR const          matrix_type,
											vk::ComponentTypeKHR const          vector_type) -> vk::Result;
private:

	vk::Device                          device;
	vk::CooperativeVectorMatrixLayoutNV layout          = vk::CooperativeVectorMatrixLayoutNV::eRowMajor;
	std::size_t                         parameters_size = 0;
};
} // namespace ng

namespace ng {

auto VulkanCoopVecNetwork::UpdateOffsetsAndSize(vk::Device                          device,
												vk::CooperativeVectorMatrixLayoutNV layout,
												vk::ComponentTypeKHR const          matrix_type,
												vk::ComponentTypeKHR const          vector_type) -> vk::Result {
	u32          current_layer_outputs = std::get<Linear>(GetLayer(0)).GetOutputsCount();
	CoopVecUtils coop_vec_utils(device);
	std::size_t  offset = 0;
	for (LayerVariant& layer : GetLayers()) {
		std::visit(
			Visitor{
				[&offset, &current_layer_outputs, this, &coop_vec_utils, layout, matrix_type, vector_type](Linear& layer) -> vk::Result {
					auto [result, size] = coop_vec_utils.CalculateByteSize(layer.GetOutputsCount(), layer.GetInputsCount(), layout, matrix_type);
					if (result != vk::Result::eSuccess) return result;
					offset = AlignTo(offset, coop_vec_utils.GetMatrixAlignment());
					layer.SetWeightsOffset(offset);
					layer.SetWeightsSize(size);
					offset += size;
					offset = AlignTo(offset, coop_vec_utils.GetVectorAlignment());
					layer.SetBiasesOffset(offset);
					layer.SetBiasesSize(layer.GetOutputsCount() * GetVulkanComponentSize(vector_type));
					offset += layer.GetBiasesSize();

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
	ng::VulkanCoopVecNetwork /* <float, float> */ network({
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
