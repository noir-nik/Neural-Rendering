
#include "BRDFSINEKAN.h"

void apply_layer_0(in float input[6], out float output[5]) {
	for (int j = 0; j < 5; j++) {
		output[j] = layer_0_bias[j];
	}

	for (int j = 0; j < 5; j++) {
		for (int k = 0; k < 6; k++) {
			for (int l = 0; l < 8; l++) {
				float s = sin(input[k] * layer_0_freq[l] + layer_0_phase[k][l]);
				output[j] += s * layer_0_amplitudes[j][k][l];
			}
		}
	}
}

void apply_layer_1(in float input[5], out float output[5]) {
	for (int j = 0; j < 5; j++) {
		output[j] = layer_1_bias[j];
	}

	for (int j = 0; j < 5; j++) {
		for (int k = 0; k < 5; k++) {
			for (int l = 0; l < 8; l++) {
				float s = sin(input[k] * layer_1_freq[l] + layer_1_phase[k][l]);
				output[j] += s * layer_1_amplitudes[j][k][l];
			}
		}
	}
}

void apply_layer_2(in float input[5], out float output[3]) {
	for (int j = 0; j < 3; j++) {
		output[j] = layer_2_bias[j];
	}

	for (int j = 0; j < 3; j++) {
		for (int k = 0; k < 5; k++) {
			for (int l = 0; l < 8; l++) {
				float s = sin(input[k] * layer_2_freq[l] + layer_2_phase[k][l]);
				output[j] += s * layer_2_amplitudes[j][k][l];
			}
		}
	}
}

void BRDFSINEKAN(in float input[6], out float output[3]) {
	// Network architecture: [6, 5, 5, 3]
	float layer_0_out[5];
	float layer_1_out[5];

	// Forward pass through all layers
	apply_layer_0(input, layer_0_out);
	apply_layer_1(layer_0_out, layer_1_out);
	apply_layer_2(layer_1_out, output);

	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
}
