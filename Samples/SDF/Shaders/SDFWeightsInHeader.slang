#include "SDFConfig.h"
#include "SDFWeights.h"
// clang-format off

func SDFWeightsInHeader<T : __BuiltinFloatingPointType>(vector<T, 3> p) -> T {
	if (length(p) > T(1.)) {
		return length(p) - T(.8);
	}

	T input0[3] = {p.x, p.y, p.z};
	T output0[16];
	T output1[16];
	T output2[16];
	T output3[1];

	for (int i = 0; i < 16; ++i) {
		output0[i] = T(kSDFBias0[i]);
		output1[i] = T(kSDFBias1[i]);
		output2[i] = T(kSDFBias2[i]);
	}
	output3[0] = T(kSDFBias3[0]);

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			output0[i] += T(kSDFWeights0[i][j]) * input0[j];
		}
		output0[i] = sin(output0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output1[i] += T(kSDFWeights1[i][j]) * output0[j];
		}
		output1[i] = sin(output1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output2[i] += T(kSDFWeights2[i][j]) * output1[j];
		}
		output2[i] = sin(output2[i]) / T(1.4) + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		output3[0] += T(kSDFWeights3[i]) * output2[i];
	}

	return output3[0];
}

func D_SDFScalar<T : __BuiltinFloatingPointType>(vector<T, 3> p) -> vector<T, 3> {
	if (length(p) > T(1.)) {
		return normalize(p);
	}

	T input0[3] = {p.x, p.y, p.z};
	T z0[16];
	T z1[16];
	T z2[16];
	T z3[1];

	for (int i = 0; i < 16; ++i) {
		z0[i] = T(kSDFBias0[i]);
		z1[i] = T(kSDFBias1[i]);
		z2[i] = T(kSDFBias2[i]);
	}
	z3[0] = T(kSDFBias3[0]);
	
	T output0[16];
	T output1[16];
	T output2[16];
	T output3[1];

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			z0[i] += T(kSDFWeights0[i][j]) * input0[j];
		}
		output0[i] = sin(z0[i]);
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			z1[i] += T(kSDFWeights1[i][j]) * output0[j];
		}
		output1[i] = sin(z1[i]) + output0[i];
	}

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			z2[i] += T(kSDFWeights2[i][j]) * output1[j];
		}
		output2[i] = sin(z2[i]) / T(1.4) + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		z3[0] += T(kSDFWeights3[i]) * output2[i];
	}
	output3 = z3;

	// Backward pass
	T d_output3 = T(1.0);
	T d_output2[16];
	T d_output1[16];
	T d_output0[16];
	T d_input[3] = {T(0), T(0), T(0)};

	for (int i = 0; i < 16; ++i) {
		d_output2[i] = d_output3 * T(kSDFWeights3[i]);
	}

	// Skip connection
	for (int i = 0; i < 16; ++i) {
		d_output1[i] = d_output2[i];
	}
	
	for (int j = 0; j < 16; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_output1[j] += d_output2[i] * T(kSDFWeights2[i][j]) * cos(z2[i]) / T(1.4);
		}
	}

	// Skip connection
	for (int i = 0; i < 16; ++i) {
		d_output0[i] = d_output1[i];
	}

	for (int j = 0; j < 16; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_output0[j] += d_output1[i] * T(kSDFWeights1[i][j]) * cos(z1[i]);
		}
	}

	for (int j = 0; j < 3; ++j) {
		for (int i = 0; i < 16; ++i) {
			d_input[j] += d_output0[i] * T(kSDFWeights0[i][j]) * cos(z0[i]);
		}
	}

	return {d_input[0], d_input[1], d_input[2]};
}
