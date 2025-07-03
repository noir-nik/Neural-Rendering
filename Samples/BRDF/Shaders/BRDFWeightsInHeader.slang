#include "BRDFConfig.h"

// #define _TY half
// #define _TY COOPVEC_TYPE
// #define WEIGHTS_SIZE_TY size_t
#include "BRDFWeights.h"
#include "BRDFWeights_split.h"

import CoopVec; //relu

// void fwd_relu<T : __BuiltinFloatingPointType>(T input[], T output[], int w_idx, int cols, int rows) {
// 	for (int i = 0; i < cols; ++i) {
// 		for (int j = 0; j < rows; ++j) {
// 			output[i] += T(kNetworkWeights[kNetworkWeights_offsets[w_idx] + i * cols + j]) * input[j];
// 			// output[i] += T(kNetworkWeights[kNetworkWeights_offsets[w_idx] + j]) * input[j];
// 		}
// 		output[i] = relu(output[i]);
// 	}
// }

vector<T, 3> BRDFWeightsInHeader1<T : __BuiltinFloatingPointType>(vector<T, 3> view, vector<T, 3> light) {

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

vector<T, 3> BRDFWeightsInHeader<T : __BuiltinFloatingPointType>(vector<T, 3> view, vector<T, 3> light) {

	T input0[6] = { view.x, view.y, view.z, light.x, light.y, light.z };
	T output0[64];
	T output1[64];
	T output2[64];
	T output3[6];
	T output4[3];

	// Preload biases
	// for (int i = 0; i < 64; ++i) {
	// 	output0[i] = T(kNetworkBias[kNetworkBias_offsets[0] + i]);
	// 	output1[i] = T(kNetworkBias[kNetworkBias_offsets[1] + i]);
	// 	output2[i] = T(kNetworkBias[kNetworkBias_offsets[2] + i]);
	// }
	// for (int i = 0; i < 6; ++i) {
	// 	output3[i] = T(kNetworkBias[kNetworkBias_offsets[3] + i]);
	// }
	// for (int i = 0; i < 3; ++i) {
	// 	output4[i] = T(kNetworkBias[kNetworkBias_offsets[4] + i]);
	// }
	// Forward pass
	{
		const int layer_idx = 0;
		const int inputs    = 6;
		const int outputs   = 64;
		for (int i = 0; i < outputs; ++i) {
			// output0[i] = T(kNetworkBias[kNetworkBias_offsets[layer_idx] + i]);
			output0[i] = T(kNetworkBias[0 + i]);
			for (int j = 0; j < inputs; ++j) {
				// output0[i] += T(kNetworkWeights[kNetworkWeights_offsets[layer_idx] + j]) * input0[j];
				// output0[i] += T(kNetworkWeights[0 + i * outputs + j]) * input0[j];
				output0[i] += T(kBRDFWeights0[i][j]) * input0[j];
			}
			output0[i] = relu(output0[i]);
		}
	}

	// for (int i = 0; i < 64; ++i) {
	// 	for (int j = 0; j < 64; ++j) {
	// 		output1[i] += T(kBRDFWeights1[i][j]) * output0[j];
	// 	}
	// 	output1[i] = relu(output1[i]);
	// }
	{
		const int layer_idx = 1;
		const int inputs    = 64;
		const int outputs   = 64;
		for (int i = 0; i < outputs; ++i) {
			// output1[i] = T(kNetworkBias[kNetworkBias_offsets[layer_idx] + i]);
			output1[i] = T(kNetworkBias[64 + i]);
			for (int j = 0; j < inputs; ++j) {
				// output1[i] += T(kNetworkWeights[kNetworkWeights_offsets[layer_idx] + j]) * output0[j];
				// output1[i] += T(kNetworkWeights[384 + i * outputs + j]) * output0[j];
				output1[i] += T(kBRDFWeights1[i][j]) * output0[j];
			}
			output1[i] = relu(output1[i]);
		}
	}

	// for (int i = 0; i < 64; ++i) {
	// 	for (int j = 0; j < 64; ++j) {
	// 		output2[i] += T(kBRDFWeights2[i][j]) * output1[j];
	// 	}
	// 	output2[i] = relu(output2[i]);
	// }
	{
		const int layer_idx = 2;
		const int inputs    = 64;
		const int outputs   = 64;
		for (int i = 0; i < outputs; ++i) {
			// output2[i] = T(kNetworkBias[kNetworkBias_offsets[layer_idx] + i]);
			output2[i] = T(kNetworkBias[128 + i]);
			for (int j = 0; j < inputs; ++j) {
				// output2[i] += T(kNetworkWeights[kNetworkWeights_offsets[layer_idx] + j]) * output1[j];
				// output2[i] += T(kNetworkWeights[4480 + i * outputs + j]) * output1[j];
				output2[i] += T(kBRDFWeights2[i][j]) * output1[j];
			}
			output2[i] = relu(output2[i]);
		}
	}

	// for (int i = 0; i < 6; ++i) {
	// 	for (int j = 0; j < 64; ++j) {
	// 		output3[i] += T(kBRDFWeights3[i][j]) * output2[j];
	// 	}
	// 	output3[i] = relu(output3[i]);
	// }
	{
		const int layer_idx = 3;
		const int inputs    = 64;
		const int outputs   = 6;
		for (int i = 0; i < outputs; ++i) {
			// output3[i] = T(kNetworkBias[kNetworkBias_offsets[layer_idx] + i]);
			output3[i] = T(kNetworkBias[192 + i]);
			for (int j = 0; j < inputs; ++j) {
				// output3[i] += T(kNetworkWeights[kNetworkWeights_offsets[layer_idx] + j]) * output2[j];
				// output3[i] += T(kNetworkWeights[8576 + i * outputs + j]) * output2[j];
				output3[i] += T(kBRDFWeights3[i][j]) * output2[j];
			}
			output3[i] = relu(output3[i]);
		}
	}

	// for (int i = 0; i < 3; ++i) {
	// 	for (int j = 0; j < 6; ++j) {
	// 		output4[i] += T(kBRDFWeights4[i][j]) * output3[j];
	// 	}
	// }

	{
		const int layer_idx = 4;
		const int inputs    = 6;
		const int outputs   = 3;
		for (int i = 0; i < outputs; ++i) {
			// output4[i] = T(kNetworkBias[kNetworkBias_offsets[layer_idx] + i]);
			output4[i] = T(kNetworkBias[198 + i]);
			for (int j = 0; j < inputs; ++j) {
				// output4[i] += T(kNetworkWeights[kNetworkWeights_offsets[layer_idx] + j]) * output3[j];
				// output4[i] += T(kNetworkWeights[8960 + i * outputs + j]) * output3[j];
				output4[i] += T(kBRDFWeights4[i][j]) * output3[j];
			}
		}
	}

	return { output4[0], output4[1], output4[2] };
}
/*
vector<T, 3> BRDFWeightsInHeader1<T : __BuiltinFloatingPointType>(vector<T, 3> view, vector<T, 3> light) {

	T input0[6] = { view.x, view.y, view.z, light.x, light.y, light.z };
	T output0[64];
	T output1[64];
	T output2[64];
	T output3[6];
	T output4[3];

	// Preload biases
	for (int i = 0; i < 64; ++i) {
		output0[i] = T(kNetworkBias[kNetworkBias_offsets[0] + i]);
		output1[i] = T(kNetworkBias[kNetworkBias_offsets[1] + i]);
		output2[i] = T(kNetworkBias[kNetworkBias_offsets[2] + i]);
	}
	for (int i = 0; i < 6; ++i) {
		output3[i] = T(kNetworkBias[kNetworkBias_offsets[3] + i]);
	}
	for (int i = 0; i < 3; ++i) {
		output4[i] = T(kNetworkBias[kNetworkBias_offsets[4] + i]);
	}
	// Forward pass

	// fwd_relu<T>(input0, output0, 0, 64, 6);
	// fwd_relu<T>(output0, output1, 1, 64, 64);
	// fwd_relu<T>(output1, output2, 2, 64, 64);
	// fwd_relu<T>(output2, output3, 3, 6, 64);
	// fwd_relu<T>(output3, output4, 4, 3, 6);

	// for (int i = 0; i < 3; ++i) {
	// 	for (int j = 0; j < 6; ++j) {
	// 		output4[i] += T(kNetworkWeights[i][j]) * output3[j];
	// 	}
	// }

	return { output4[0], output4[1], output4[2] };
}


 */

