void SDFWeightsInBuffer<T : __BuiltinFloatingPointType>(in T input[3], out T output[1], StructuredBuffer<T> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4]) {
	vector<T, 3> p = { input[0], input[1], input[2] };
	if (length(p) > T(1.)) {
		// return length(p) - T(.8);
		output[0] = length(p) - T(.8);
		return;
	}

	T input0[3] = { p.x, p.y, p.z };
	T output0[16];
	T output1[16];
	T output2[16];
	T output3[1];


	for (int i = 0; i < 16; ++i) {
		output0[i] = sdf_weights[bias_offsets[0] / sizeof(T) + i];
		output1[i] = sdf_weights[bias_offsets[1] / sizeof(T) + i];
		output2[i] = sdf_weights[bias_offsets[2] / sizeof(T) + i];
	}
	output3[0] = sdf_weights[bias_offsets[3] / sizeof(T)];

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			output0[i] += sdf_weights[weights_offsets[0] / sizeof(T) + i * 3 + j] * input0[j];
		}
		output0[i] = sin(output0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output1[i] += sdf_weights[weights_offsets[1] / sizeof(T) + i * 16 + j] * output0[j];
		}
		output1[i] = sin(output1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output2[i] += sdf_weights[weights_offsets[2] / sizeof(T) + i * 16 + j] * output1[j];
		}
		output2[i] = sin(output2[i]) / T(1.4) + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		output3[0] += sdf_weights[weights_offsets[3] / sizeof(T) + i] * output2[i];
	}

	// return output3[0];
	output[0] = output3[0];
}


