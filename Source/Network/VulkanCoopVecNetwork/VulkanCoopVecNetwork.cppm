export module NeuralGraphics:VulkanCoopVecNetwork;
import :GenericNetwork;
import :Core;
import :Utils;
import std;
import vulkan_hpp;

struct float16_t;
template <typename T>
constexpr inline auto GetVulkanComponentType() -> vk::ComponentTypeKHR {
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

export template <typename T, typename U>
constexpr inline auto AlignUp(T const value, U const alignment) -> T {
	return ((value + alignment - T(1)) / alignment) * alignment;
}

export template <typename T, typename U>
constexpr inline auto AlignUpPowerOfTwo(T const value, U const alignment) -> T {
	return (value + alignment - T(1)) & ~(alignment - T(1));
}

export class VulkanCoopVecNetwork : public GenericNetwork {
public:
	VulkanCoopVecNetwork(std::initializer_list<LayerVariant> layers) : GenericNetwork(layers) {};
	// VulkanCoopVecNetwork(std::span<LayerVariant> layers) : GenericNetwork(layers) {};
	auto GetParametersSize() const -> std::size_t { return parameters_size; }
	void Print();
	auto PrintParameters(std::byte const* parameters) -> void;
	auto PrintLayerWeights(int layer_index, vk::ComponentTypeKHR component_type, std::byte const* parameters) -> void;
	auto PrintLayerBiases(int layer_index, vk::ComponentTypeKHR component_type, std::byte const* parameters) -> void;

	[[nodiscard]] auto UpdateOffsetsAndSize(
		vk::Device                          device,
		vk::CooperativeVectorMatrixLayoutNV layout,
		vk::ComponentTypeKHR const          matrix_type,
		vk::ComponentTypeKHR const          vector_type) -> vk::Result;

private:
	vk::CooperativeVectorMatrixLayoutNV layout      = vk::CooperativeVectorMatrixLayoutNV::eRowMajor;
	vk::ComponentTypeKHR                matrix_type = vk::ComponentTypeKHR::eFloat32;
	vk::ComponentTypeKHR                vector_type = vk::ComponentTypeKHR::eFloat32;

	vk::Device  device;
	std::size_t parameters_size = 0;
};
