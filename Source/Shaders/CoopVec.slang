CoopVec<T, M> LinearForward<T : __BuiltinFloatingPointType, int M, int K>(
	CoopVec<T, K>                  ip,
	StructuredBuffer<T>            matrixBiasBuffer,
	uint                           matrixOffset,
	int                            biasOffset,
	constexpr CoopVecMatrixLayout  matrixLayout,
	constexpr CoopVecComponentType componentType,
	bool                           transpose = false) {
	return coopVecMatMulAdd<T, M>(
		ip,
		componentType,
		matrixBiasBuffer,
		matrixOffset,
		componentType,
		matrixBiasBuffer,
		biasOffset,
		componentType,
		matrixLayout,
		false, // transpose
		0);
}

constexpr func CoopVecComponentFromType<T>()->CoopVecComponentType {
	if (T is float16_t || T is half) {
		return CoopVecComponentType::Float16;
	} else if (T is float32_t) {
		return CoopVecComponentType::Float32;
	} else if (T is float64_t) {
		return CoopVecComponentType::Float64;
	} else if (T is int8_t) {
		return CoopVecComponentType::SignedInt8;
	} else if (T is int16_t) {
		return CoopVecComponentType::SignedInt16;
	} else if (T is int32_t) {
		return CoopVecComponentType::SignedInt32;
	} else if (T is int64_t) {
		return CoopVecComponentType::SignedInt64;
	} else if (T is uint8_t) {
		return CoopVecComponentType::UnsignedInt8;
	} else if (T is uint16_t) {
		return CoopVecComponentType::UnsignedInt16;
	} else if (T is uint32_t) {
		return CoopVecComponentType::UnsignedInt32;
	} else if (T is uint64_t) {
		return CoopVecComponentType::UnsignedInt64;
	}

	// static_assert(false, "Unsupported type");
	return CoopVecComponentType::Float32;
}

func sin<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)->CoopVec<T, N> {
	CoopVec<T, N> output;
	[ForceUnroll]
	for (int i = 0; i < N; ++i) {
		output[i] = sin(input[i]);
	}
	return output;
}

// public
func cos<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)->CoopVec<T, N> {
	CoopVec<T, N> output;
	[ForceUnroll]
	for (int i = 0; i < N; ++i) {
		output[i] = cos(input[i]);
	}
	return output;
}

func relu<T : __BuiltinFloatingPointType>(T x)->T {
	return max(x, T(0.));
}

// func relu<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)->CoopVec<T, N> {
// 	CoopVec<T, N> output;
// 	[ForceUnroll]
// 	for (int i = 0; i < N; ++i) {
// 		output[i] = relu(input[i]);
// 	}
// 	return output;
// }

func relu<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input)->CoopVec<T, N> {
	return max(input, CoopVec<T, N>(T(0.)));
}

