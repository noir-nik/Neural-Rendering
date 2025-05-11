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
	parameters_size = offset;
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
