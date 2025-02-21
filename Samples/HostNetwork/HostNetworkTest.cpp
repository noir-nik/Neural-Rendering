import NeuralGraphics;
import std;

int main(int argc, char* argv[]) {
	ng::HostNetwork<float> model({
		ng::Linear(28 * 28, 100),
		ng::Relu(),
		ng::Linear(100, 10),
	});

	std::vector<float> scratchBuffer(model.GetScratchBufferSize());
	std::vector<float> gradientBuffer(model.GetParametersSize());

	std::vector<float> lossGradients(model.GetLayers().back().GetOutputSize());

	model.Forward({}, {});

	std::printf("%zu\n", model.GetLayers().size());

	auto loss      = ng::Loss::MeanSquaredError();
	auto optimizer = ng::Optimizer::SGD(0.01);

	for (auto epoch : std::views::iota(0) | std::views::take(10)) {
		for (auto i : std::views::iota(0) | std::views::take(60000)) {
			auto result    = model.Forward(input[i]);
			float lossValue = loss(result, targets[i]);
			model.Backward( gradientBuffer.data());
			optimizer.Step(model.GetParameters(), gradientBuffer.data());
		}

		auto result    = model.Forward(input[0]);
		auto lossValue = loss.Calculate(result, output[0]);
		std::printf("Epoch %d, Loss: %f\n", epoch, lossValue);
	}
}
