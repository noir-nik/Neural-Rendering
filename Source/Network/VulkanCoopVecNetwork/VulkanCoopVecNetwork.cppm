export module NeuralGraphics:VulkanCoopVecNetwork;
import :GenericNetwork;
import :Core;
import :Utils;
import std;
import vulkan_hpp;

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

	// Total parameters in destination buffer, possibly with gaps due to alignment
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

	auto GetLayout() const -> vk::CooperativeVectorMatrixLayoutNV { return layout; };
	auto GetMatrixType() const -> vk::ComponentTypeKHR { return matrix_type; };
	auto GetVectorType() const -> vk::ComponentTypeKHR { return vector_type; };

private:
	vk::CooperativeVectorMatrixLayoutNV layout      = vk::CooperativeVectorMatrixLayoutNV::eRowMajor;
	vk::ComponentTypeKHR                matrix_type = vk::ComponentTypeKHR::eFloat32;
	vk::ComponentTypeKHR                vector_type = vk::ComponentTypeKHR::eFloat32;

	vk::Device  device;
	std::size_t parameters_size = 0;
};
