#include "SDFWeights.h"
// clang-format off

func SDFScalar(float3 p) -> float {
	if(length(p) > 1.) {
		return length(p) - .8;
	}

	float input0[3] = {p.x, p.y, p.z};
	float output0[16] = kSDFBias0;
	float output1[16] = kSDFBias1;
	float output2[16] = kSDFBias2;
	float output3[1] = kSDFBias3;

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			output0[i] += kSDFWeights0[i][j] * input0[j];
		}
		output0[i] = sin(output0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output1[i] += kSDFWeights1[i][j] * output0[j];
		}
		output1[i] = sin(output1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output2[i] += kSDFWeights2[i][j] * output1[j];
		}
		output2[i] = sin(output2[i]) / 1.4 + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		output3[0] += kSDFWeights3[i] * output2[i];
	}

	return output3[0];
}

func D_SDFScalar(float3 p) -> float3 {
	if (length(p) > 1.) {
		return normalize(p);
	}

	float input0[3] = {p.x, p.y, p.z};
	float z0[16] = kSDFBias0;
	float z1[16] = kSDFBias1;
	float z2[16] = kSDFBias2;
	float z3[1] = kSDFBias3;
	
	float output0[16];
	float output1[16];
	float output2[16];
	float output3[1];

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			z0[i] += kSDFWeights0[i][j] * input0[j];
		}
		output0[i] = sin(z0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			z1[i] += kSDFWeights1[i][j] * output0[j];
		}
		output1[i] = sin(z1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			z2[i] += kSDFWeights2[i][j] * output1[j];
		}
		output2[i] = sin(z2[i]) / 1.4 + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		z3[0] += kSDFWeights3[i] * output2[i];
	}
	output3 = z3;

	// Backward pass
	float d_output3 = 1.0f;
	float d_output2[16];
	float d_output1[16];
	float d_output0[16];
	float d_input[3] = {0, 0, 0};

	for (int i = 0; i < 16; ++i) {
		d_output2[i] = d_output3 * kSDFWeights3[i];
	}

	// Skip connection
	for (int i = 0; i < 16; ++i) {
		d_output1[i] = d_output2[i];
	}
	
	for (int j = 0; j < 16; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_output1[j] += d_output2[i] * kSDFWeights2[i][j] * cos(z2[i]) / 1.4;
		}
	}

	// Skip connection
	for (int i = 0; i < 16; ++i) {
		d_output0[i] = d_output1[i];
	}

	for (int j = 0; j < 16; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_output0[j] += d_output1[i] * kSDFWeights1[i][j] * cos(z1[i]);
		}
	}

	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_input[j] += d_output0[i] * kSDFWeights0[i][j] * cos(z0[i]);
		}
	}

	return {d_input[0], d_input[1], d_input[2]};
}
