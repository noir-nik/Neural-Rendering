
#include "BRDFSINEKAN.h"

void BRDFSINEKAN(in float input[6], out float output[3]) {
	float layer_0_out[layer_0_bias_shape0];
	float layer_1_out[layer_1_bias_shape0];

	// Layer 0
	for (int j = 0; j < layer_0_bias_shape0; j++) {
		layer_0_out[j] = layer_0_bias[j];
	}

	for (int j = 0; j < layer_0_amplitudes_shape0; j++) {
		for (int k = 0; k < layer_0_amplitudes_shape1; k++) {
			for (int l = 0; l < layer_0_amplitudes_shape2; l++) {
				float s = sin(input[k] * layer_0_freq[l] + layer_0_phase[k][l]);
				layer_0_out[j] += s * layer_0_amplitudes[j][k][l];
			}
		}
	}

	// Layer 1
	for (int j = 0; j < layer_1_bias_shape0; j++) {
		layer_1_out[j] = layer_1_bias[j];
	}

	for (int j = 0; j < layer_1_amplitudes_shape0; j++) {
		for (int k = 0; k < layer_1_amplitudes_shape1; k++) {
			for (int l = 0; l < layer_1_amplitudes_shape2; l++) {
				float s = sin(layer_0_out[k] * layer_1_freq[l] + layer_1_phase[k][l]);
				layer_1_out[j] += s * layer_1_amplitudes[j][k][l];
			}
		}
	}

	// Layer 2
	for (int j = 0; j < layer_2_bias_shape0; j++) {
		output[j] = layer_2_bias[j];
	}

	for (int j = 0; j < layer_2_amplitudes_shape0; j++) {
		for (int k = 0; k < layer_2_amplitudes_shape1; k++) {
			for (int l = 0; l < layer_2_amplitudes_shape2; l++) {
				float s = sin(layer_1_out[k] * layer_2_freq[l] + layer_2_phase[k][l]);
				output[j] += s * layer_2_amplitudes[j][k][l];
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
}
