#include "BRDFConfig.h"

// #define _TY half
// #define _TY COOPVEC_TYPE
// #define WEIGHTS_SIZE_TY size_t
#include "BRDFWeights.h"

import CoopVec; //relu

vector<T, 3> BRDFWeightsInHeader<T : __BuiltinFloatingPointType>(vector<T, 3> view, vector<T, 3> light) {

	T input0[6] = { view.x, view.y, view.z, light.x, light.y, light.z };
	T output0[64];
	T output1[64];
	T output2[64];
	T output3[6];
	T output4[3];

	// Preload biases
	for (int i = 0; i < 64; ++i) {
		output0[i] = T(kBRDFBias0[i]);
		output1[i] = T(kBRDFBias1[i]);
		output2[i] = T(kBRDFBias2[i]);
	}
	for (int i = 0; i < 6; ++i) {
		output3[i] = T(kBRDFBias3[i]);
	}
	for (int i = 0; i < 3; ++i) {
		output4[i] = T(kBRDFBias4[i]);
	}
	// Forward pass
	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 6; ++j) {
			output0[i] += T(kBRDFWeights0[i][j]) * input0[j];
		}
		output0[i] = relu(output0[i]);
	}

	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 64; ++j) {
			output1[i] += T(kBRDFWeights1[i][j]) * output0[j];
		}
		output1[i] = relu(output1[i]);
	}

	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 64; ++j) {
			output2[i] += T(kBRDFWeights2[i][j]) * output1[j];
		}
		output2[i] = relu(output2[i]);
	}

	for (int i = 0; i < 6; ++i) {
		for (int j = 0; j < 64; ++j) {
			output3[i] += T(kBRDFWeights3[i][j]) * output2[j];
		}
		output3[i] = relu(output3[i]);
	}

	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 6; ++j) {
			output4[i] += T(kBRDFWeights4[i][j]) * output3[j];
		}
	}

	return { output4[0], output4[1], output4[2] };
}


