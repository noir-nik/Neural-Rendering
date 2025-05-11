module NeuralGraphics;
import :VulkanCoopVecNetwork;
import vulkan_hpp;
import :Utils;

import std;

using namespace Utils;

auto VulkanCoopVecNetwork::UpdateOffsetsAndSize(vk::Device                          device,
												vk::CooperativeVectorMatrixLayoutNV layout,
												vk::ComponentTypeKHR const          matrix_type,
												vk::ComponentTypeKHR const          vector_type) -> vk::Result {
	u32 current_layer_outputs = std::get<Linear>(GetLayer(0)).GetOutputsCount();

	std::size_t offset = 0;
	for (LayerVariant& layer : GetLayers()) {
		std::visit(
			Visitor{
				[&offset, &current_layer_outputs, this, device, layout, matrix_type, vector_type](Linear& layer) -> vk::Result {
					auto [result, size] = CoopVecUtils::CalculateByteSize(device, layer.GetOutputsCount(), layer.GetInputsCount(), layout, matrix_type);
					if (result != vk::Result::eSuccess) return result;
					offset = AlignUpPowerOfTwo(offset, CoopVecUtils::GetMatrixAlignment());
					layer.SetWeightsOffset(offset);
					layer.SetWeightsSize(size);
					offset += size;
					offset = AlignUpPowerOfTwo(offset, CoopVecUtils::GetVectorAlignment());
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
	parameters_size   = offset;
	this->layout      = layout;
	this->matrix_type = matrix_type;
	this->vector_type = vector_type;
	return vk::Result::eSuccess;
}

auto VulkanCoopVecNetwork::Print() -> void {
	std::printf("+--------+----------+----------+----------+----------+----------+----------+----------+----------+\n");
	std::printf("| Layer  | Inputs   | Outputs  | Weights  | Biases   | Params   | Weights  | Biases   | Params   |\n");
	std::printf("|        | Count    | Count    | Count    | Count    | Count    | Offset   | Offset   | Size     |\n");
	std::printf("+--------+----------+----------+----------+----------+----------+----------+----------+----------+\n");

	u32 counter = 0;
	for (auto& layer : GetLayers()) {
		std::visit(
			Visitor{
				[&counter](Linear const& layer) {
					std::printf("| %6u | %8u | %8u | %8u | %8u | %8u | %8zu | %8zu | %8zu |\n",
								counter, layer.GetInputsCount(), layer.GetOutputsCount(),
								layer.GetWeightsCount(), layer.GetBiasesCount(), layer.GetParametersCount(),
								layer.GetWeightsOffset(), layer.GetBiasesOffset(), layer.GetParametersSize());
					++counter;
				},
				[counter](auto const&) {}},
			layer);
	}
	std::printf("+--------+----------+----------+----------+----------+----------+----------+----------+----------+\n");
}
auto VulkanCoopVecNetwork::PrintLayerWeights(int layer_index, vk::ComponentTypeKHR component_type, std::byte const* parameters) -> void {
	if (!parameters) return;
	if (matrix_type != vk::ComponentTypeKHR::eFloat32) return;
	if (GetLayer(layer_index).Is<Linear>()) {
		auto const& layer    = GetLayer<Linear>(layer_index);
		auto const* p_weight = reinterpret_cast<float const*>(parameters + layer.GetWeightsOffset());

		for (u32 i = 0; i < layer.GetOutputsCount(); ++i) {
			for (u32 j = 0; j < layer.GetInputsCount(); ++j) {
				std::printf("%10.6f ", p_weight[i * layer.GetInputsCount() + j]);
			}
			std::printf("\n");
		}
	}
}

auto VulkanCoopVecNetwork::PrintLayerBiases(int layer_index, vk::ComponentTypeKHR component_type, std::byte const* parameters) -> void {
	if (!parameters) return;
	if (vector_type != vk::ComponentTypeKHR::eFloat32) return;
	if (GetLayer(layer_index).Is<Linear>()) {
		auto const& layer  = GetLayer<Linear>(layer_index);
		auto const* p_bias = reinterpret_cast<float const*>(parameters + layer.GetBiasesOffset());

		for (u32 i = 0; i < layer.GetBiasesCount(); ++i) {
			std::printf("%10.6f ", p_bias[i]);
		}
		std::printf("\n");
	}
}

auto VulkanCoopVecNetwork::PrintParameters(std::byte const* parameters) -> void {
	if (!parameters) return;
	if (matrix_type != vk::ComponentTypeKHR::eFloat32 || vector_type != vk::ComponentTypeKHR::eFloat32) return;
	for (int layer_index = 0; layer_index < GetLayers().size(); ++layer_index) {
		if (GetLayer(layer_index).Is<Linear>()) {
			auto const& layer    = GetLayer<Linear>(layer_index);
			auto const* p_weight = reinterpret_cast<float const*>(parameters + layer.GetWeightsOffset());
			auto const* p_bias   = reinterpret_cast<float const*>(parameters + layer.GetBiasesOffset());

			std::printf("Layer %d weights:\n", layer_index);
			PrintLayerWeights(layer_index, matrix_type, parameters);
			std::printf("Layer %d biases:\n", layer_index);
			PrintLayerBiases(layer_index, vector_type, parameters);
		}
	}
}
