export module NeuralGraphics:GenericLayers;
import :Core;
import :Utils;
import std;

using u32 = std::uint32_t;
using Utils::IsAnyV;

export class ILayer {
public:
	ILayer(u32 inputs_count, u32 output_count) : input_count(inputs_count), output_count(output_count) {}
	virtual ~ILayer() {}
	constexpr virtual auto GetInputsCount() const -> u32 { return input_count; }
	constexpr virtual auto GetOutputsCount() const -> u32 { return output_count; }
	constexpr virtual void SetInputsCount(u32 size) { input_count = size; }
	constexpr virtual void SetOutputsCount(u32 size) { output_count = size; }

private:
	u32 input_count;
	u32 output_count;
};

export class Linear : public ILayer {
public:
	Linear(u32 input_size, u32 output_size) : ILayer(input_size, output_size) {}

	auto GetWeightsCount() const -> u32 { return GetInputsCount() * GetOutputsCount(); }
	auto GetBiasesCount() const -> u32 { return GetOutputsCount(); }
	auto GetParametersCount() const -> u32 { return GetWeightsCount() + GetBiasesCount(); }

	// Get offset in bytes
	auto GetWeightsOffset() const -> std::size_t { return weights_offset; }
	auto GetBiasesOffset() const -> std::size_t { return bias_offset; }
	auto GetWeightsSize() const -> std::size_t { return weights_size; }
	auto GetBiasesSize() const -> std::size_t { return bias_size; }
	auto GetParametersSize() const -> std::size_t { return weights_size + bias_size; }

	// Set offset in bytes
	void SetWeightsOffset(std::size_t offset) { weights_offset = offset; }
	void SetBiasesOffset(std::size_t offset) { bias_offset = offset; }
	void SetWeightsSize(std::size_t size) { weights_size = size; }
	void SetBiasesSize(std::size_t size) { bias_size = size; }

private:
	std::size_t weights_offset = 0;
	std::size_t weights_size = 0; // size in bytes
	std::size_t bias_offset = 0;
	std::size_t bias_size = 0;
};

export class Relu : public ILayer {
public:
	Relu(u32 size = 0) : ILayer(size, size) {}

	auto GetParametersCount() const -> u32 { return 0; }
};

export class Sigmoid : public ILayer {
public:
	Sigmoid(u32 size = 0) : ILayer(size, size) {}
};

export class Softmax : public ILayer {
public:
	Softmax(u32 size = 0) : ILayer(size, size) {}
};

export class Sin : public ILayer {
public:
	Sin(u32 size = 0) : ILayer(size, size) {}
};


template <typename T>
concept GenericLayerType = IsAnyV<T, Linear, Relu, Sigmoid, Softmax, Sin>;

export using LayerVariantBase = std::variant<Linear, Relu, Sigmoid, Softmax, Sin>;

export struct LayerVariant : LayerVariantBase {
	using LayerVariantBase::LayerVariantBase;

	auto GetOutputsCount() const -> u32 {
		return std::visit([](auto const& layer) { return layer.GetOutputsCount(); }, *this);
	}
	auto GetInputsCount() const -> u32 {
		return std::visit([](auto const& layer) { return layer.GetInputsCount(); }, *this);
	}

	template <GenericLayerType T>
	constexpr auto Get() -> T& { return std::get<T>(*this); }
};

export template <typename T>
struct IsActivationLayer {
	static constexpr bool value = IsAnyV<T, Relu, Sigmoid, Softmax, Sin>;
};

export template <typename T>
inline constexpr bool IsActivationLayerV = IsActivationLayer<T>::value;

