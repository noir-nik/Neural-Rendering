func SDFScalarBuffer(float3 p, StructuredBuffer<float> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4]) -> float {
	if(length(p) > 1.) {
		return length(p) - .8;
	}

	float input0[3] = {p.x, p.y, p.z};
	float output0[16];
	float output1[16];
	float output2[16];
	float output3[1];

	for (int i = 0; i < 16; ++i) {
		output0[i] = sdf_weights[bias_offsets[0]/4 + i];
		output1[i] = sdf_weights[bias_offsets[1]/4 + i];
		output2[i] = sdf_weights[bias_offsets[2]/4 + i];
	}
	output3[0] = sdf_weights[bias_offsets[3]/4];

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			output0[i] += sdf_weights[weights_offsets[0]/4 + i * 3 + j] * input0[j];
		}
		output0[i] = sin(output0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output1[i] += sdf_weights[weights_offsets[1]/4 + i * 16 + j] * output0[j];
		}
		output1[i] = sin(output1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output2[i] += sdf_weights[weights_offsets[2]/4 + i * 16 + j] * output1[j];
		}
		output2[i] = sin(output2[i]) / 1.4 + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		output3[0] += sdf_weights[weights_offsets[3]/4 + i] * output2[i];
	}

	return output3[0];
}