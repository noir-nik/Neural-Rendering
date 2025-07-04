import CoopVec;

void BRDFWeightsInBuffer<T : __BuiltinFloatingPointType>(in T input[6], out T output[3], StructuredBuffer<T> weights, uint32_t weights_offsets[5], uint32_t bias_offsets[5]) {
 
	T output0[64];
	T output1[64];
	T output2[64];
	T output3[6];
	// T output[3];

	// Preload biases
	for (int i = 0; i < 64; ++i) {
		output0[i] = weights[bias_offsets[0] / sizeof(T) + i];
		output1[i] = weights[bias_offsets[1] / sizeof(T) + i];
		output2[i] = weights[bias_offsets[2] / sizeof(T) + i];
	}
	for (int i = 0; i < 6; ++i) {
		output3[i] = weights[bias_offsets[3] / sizeof(T) + i];
	}
	for (int i = 0; i < 3; ++i) {
		output[i] = weights[bias_offsets[4] / sizeof(T) + i];
	}

	// Forward pass
	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 6; ++j) {
			output0[i] += weights[weights_offsets[0] / sizeof(T) + i * 6 + j] * input[j];
		}
		output0[i] = relu(output0[i]);
	}

	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 64; ++j) {
			output1[i] += weights[weights_offsets[1] / sizeof(T) + i * 64 + j] * output0[j];
		}
		output1[i] = relu(output1[i]);
	}

	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 64; ++j) {
			output2[i] += weights[weights_offsets[2] / sizeof(T) + i * 64 + j] * output1[j];
		}
		output2[i] = relu(output2[i]);
	}

	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 64; ++j) {
			output3[i] += weights[weights_offsets[3] / sizeof(T) + i * 64 + j] * output2[j];
		}
		output3[i] = relu(output3[i]);
	}

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 6; ++j) {
			output[i] += weights[weights_offsets[4] / sizeof(T) + i * 6 + j] * output3[j];
		}
	}

	// return { output4[0], output4[1], output4[2] };
}


