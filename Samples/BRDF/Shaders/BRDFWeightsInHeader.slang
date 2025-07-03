// #include "BRDFConfig.h"

// #define _TY half
// #define _TY COOPVEC_TYPE
// #define WEIGHTS_SIZE_TY size_t
// #include "BRDFWeights.h"
#include "BRDFWeights_split.h"

import CoopVec; //relu

void BRDFWeightsInHeader<T : __BuiltinFloatingPointType>(in T input[6], out T output[3]) {
	T a[128];
	for (int i = 0; i < 64; ++i) {
		a[i] = T(kBRDFBias0[i]);
		for (int j = 0; j < 6; ++j) {
			a[i] += T(kBRDFWeights0[i][j]) * input[j];
		}
		a[i] = relu(a[i]);
	}
	for (int i = 0; i < 64; ++i) {
		a[i + 64] = T(kBRDFBias1[i]);
		for (int j = 0; j < 64; ++j) {
			a[i + 64] += T(kBRDFWeights1[i][j]) * a[j];
		}
		a[i + 64] = relu(a[i + 64]);
	}
	for (int i = 0; i < 64; ++i) {
		a[i] = T(kBRDFBias2[i]);
		for (int j = 0; j < 64; ++j) {
			a[i] += T(kBRDFWeights2[i][j]) * a[j + 64];
		}
		a[i] = relu(a[i]);
	}
	for (int i = 0; i < 6; ++i) {
		a[i + 64] = T(kBRDFBias3[i]);
		for (int j = 0; j < 64; ++j) {
			a[i + 64] += T(kBRDFWeights3[i][j]) * a[j];
		}
		a[i + 64] = relu(a[i + 64]);
	}
	for (int i = 0; i < 3; ++i) {
		output[i] = T(kBRDFBias4[i]);
		for (int j = 0; j < 6; ++j) {
			output[i] += T(kBRDFWeights4[i][j]) * a[j + 64];
		}
	}
}


