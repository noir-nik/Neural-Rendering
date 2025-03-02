export module NeuralGraphics:HostLayers;
import :GenericLayers;
import :Core;
import std;

export namespace ng {

template <typename T>
class HostLinear : public Linear {
public:
	HostLinear(Linear const& layer) : Linear(layer) {}
	void Forward(T const* inputs, T* outputs, T* weights, T* bias);
	void Backward(T const* inputs, T const* gradOutputs, T* gradWeights, T* gradBiases, T* weights, T* bias);
};

template <typename T>
class HostRelu : public Relu {
public:
	HostRelu(Relu const& layer) : Relu(layer) {}
	void Forward(T const* inputs, T* outputs);
	void Backward(T const* inputs, T* outputs);
};

template <typename T>
class HostSigmoid : public Sigmoid {
public:
	HostSigmoid(Sigmoid const& layer) : Sigmoid(layer) {}
	void Forward(T const* inputs, T* outputs);
	void Backward(T const* inputs, T* outputs);
};

template <typename T>
class HostSoftmax : public Softmax {
public:
	HostSoftmax(Softmax const& layer) : Softmax(layer) {}
	void Forward(T const* inputs, T* outputs);
	void Backward(T const* forwardOutputs, T* gradOutputs);
};

template <typename T>
class HostSin : public Sin {
public:
	HostSin(Sin const& layer) : Sin(layer) {}
	void Forward(T const* inputs, T* outputs);
	void Backward(T const* forwardOutputs, T* gradOutputs);
};


} // namespace ng


export namespace ng {

// MatMul
// C = A * B
// A: M x K
// B: K x N
// C: M x N
template <typename T>
void MatMul(T const* A, T const* B, T* result, u32 const M, u32 const K, u32 const N) {
	for (u32 i = 0; i < M; ++i) {
		for (u32 j = 0; j < N; ++j) {
			result[i * N + j] = 0;
			for (u32 k = 0; k < K; ++k) {
				result[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}

template <typename T>
void MatMulAdd(T const* A, T const* B, T const* C, T* result, u32 const M, u32 const K, u32 const N) {
	for (u32 i = 0; i < M; ++i) {
		for (u32 j = 0; j < N; ++j) {
			result[i * N + j] = C[i * N + j];
			for (u32 k = 0; k < K; ++k) {
				result[i * N + j] += A[i * K + k] * B[k * N + j];
			}
		}
	}
}

// B is transposed
template<typename T>
void MatMulTransposed(T const* A, T const* B, T* result, u32 const M, u32 const K, u32 const N) {
	for (u32 i = 0; i < M; ++i) {
		for (u32 j = 0; j < N; ++j) {
			result[i * N + j] = 0;
			for (u32 k = 0; k < K; ++k) {
				result[i * N + j] += A[i * K + k] * B[j * K + k];
			}
		}
	}
}

template <typename T>
void MatMulVec(T const* matrix, T const* vector, T* result, u32 const M, u32 const K) {
	for (u32 i = 0; i < M; ++i) {
		result[i] = 0;
		for (u32 k = 0; k < K; ++k) {
			result[i] += matrix[i * K + k] * vector[k];
		}
	}
}

template <typename T>
void MatMulVecAdd(T const* matrix, T const* vector, T const* bias, T* result, u32 const M, u32 const K) {
	for (u32 i = 0; i < M; ++i) {
		result[i] = bias[i];
		for (u32 k = 0; k < K; ++k) {
			result[i] += matrix[i * K + k] * vector[k];
		}
	}
}

template <typename T>
void MatMulVecTransposed(T const* matrix, T const* vector, T* result, u32 const M, u32 const K) {
	for (u32 i = 0; i < M; ++i) {
		result[i] = 0;
		for (u32 k = 0; k < K; ++k) {
			result[i] += matrix[k * M + i] * vector[k];
		}
	}
}


template <typename T>
void HostLinear<T>::Forward(T const* inputs, T* outputs, T* weights, T* bias) {
	MatMulVecAdd(weights, inputs, bias, outputs, GetOutputsCount(), GetInputsCount());
}

template <typename T>
void HostLinear<T>::Backward(T const* inputs, T const* gradOutputs, T* gradWeights, T* gradBiases, T* weights, T* bias) {
	// NOT IMPLEMENTED
	// MatMulVecTransposed(weights, inputs, gradWeights, GetInputsCount(), GetOutputsCount());
	// MatMulVecTransposed(inputs, gradOutputs, gradBiases, 1, GetInputsCount());
}

template <typename T>
void HostRelu<T>::Forward(T const* inputs, T* outputs) {
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] = inputs[i] > 0 ? inputs[i] : 0;
	}
}

template <typename T>
void HostRelu<T>::Backward(T const* inputs, T* outputs) {
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] = inputs[i] > 0 ? 1 : 0;
	}
}

template <typename T>
void HostSigmoid<T>::Forward(T const* inputs, T* outputs) {
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] = T{1} / (T{1} + std::exp(-inputs[i]));
	}
}

template <typename T>
void HostSigmoid<T>::Backward(T const* inputs, T* outputs) {
	auto sigmoid = [](T x) { return T{1} / (T{1} + std::exp(-x)); };
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] = sigmoid(inputs[i]) * (T{1} - sigmoid(inputs[i]));
	}
}

template <typename T>
void HostSoftmax<T>::Forward(T const* inputs, T* outputs) {
	T max = -std::numeric_limits<T>::infinity();
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		max = std::max(max, inputs[i]);
	}
	T sum = 0;
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		T exp_input = std::exp(inputs[i] - max);
		sum += exp_input;
		outputs[i] = exp_input;
	}
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] /= sum;
	}
}

template <typename T>
void HostSoftmax<T>::Backward(T const* forwardOutputs, T* gradOutputs) {
	// NOT IMPLEMENTED
	T sum = 0;
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		sum += forwardOutputs[i] * gradOutputs[i];
	}
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		T grad = forwardOutputs[i] * (gradOutputs[i] - sum);
		gradOutputs[i] = grad;
	}
}

template <typename T>
void HostSin<T>::Forward(T const* inputs, T* outputs) {
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		outputs[i] = std::sin(inputs[i]);
	}
}

template <typename T>
void HostSin<T>::Backward(T const* forwardOutputs, T* gradOutputs) {
	for (u32 i = 0; i < GetOutputsCount(); ++i) {
		gradOutputs[i] = std::cos(forwardOutputs[i]);
	}
}

} // namespace ng
