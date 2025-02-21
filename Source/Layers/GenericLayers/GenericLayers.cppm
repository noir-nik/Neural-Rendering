export module NeuralGraphics:GenericLayers;
import :Core;
import :Util;
import std;

export namespace ng {

class ILayer {
public:
	ILayer(u32 inputSize, u32 outputSize) : inputSize(inputSize), outputSize(outputSize) {}
	virtual ~ILayer() {}
	virtual auto GetInputSize() const -> u32 { return inputSize; }
	virtual auto GetOutputSize() const -> u32 { return outputSize; }
	virtual void SetInputSize(u32 size) { inputSize = size; }
	virtual void SetOutputSize(u32 size) { outputSize = size; }
private:
	u32 inputSize;
	u32 outputSize;
};

class Linear : public ILayer {
public:
	Linear(u32 inputSize, u32 outputSize) : ILayer(inputSize, outputSize) {}

	auto GetWeightsSize() const -> u32 { return GetInputSize() * GetOutputSize(); }
	auto GetBiasesSize() const -> u32 { return GetOutputSize(); }
	auto GetParametersSize() const -> u32 { return GetWeightsSize() + GetBiasesSize(); }

	auto GetWeightsOffset() const -> u32 { return weightsOffset; }
	auto GetBiasesOffset() const -> u32 { return biasOffset; }

	void SetWeightsOffset(u32 offset) { weightsOffset = offset; }
	void SetBiasesOffset(u32 offset) { biasOffset = offset; }

private:
	u32 weightsOffset;
	u32 biasOffset;
};

class Relu : public ILayer {
public:
	Relu(u32 size = 0) : ILayer(size, size) {}
};

class Sigmoid : public ILayer {
public:
	Sigmoid(u32 size = 0) : ILayer(size, size) {}
};

class Softmax : public ILayer {
public:
	Softmax(u32 size = 0) : ILayer(size, size) {}
};

using LayerVariant = std::variant<Linear, Relu, Sigmoid, Softmax>;

template <typename T>
struct IsActivationLayer {
	static constexpr bool value = IsAnyV<T, Relu, Sigmoid, Softmax>;
};

template <typename T>
inline constexpr bool IsActivationLayerV = IsActivationLayer<T>::value;


} // namespace ng
