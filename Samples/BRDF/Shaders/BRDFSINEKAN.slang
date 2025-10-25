
// #include "BRDFSINEKAN.h"
// #include "SINEKAN_2_2_3_82_112.h"

// import SINEKAN_2_2_3_82_112;

#ifndef INC
// #define INC SINEKAN_2_2_3_82_112
#include "BRDFSINEKAN.h"
#else
import INC;
#endif


/* void BRDFSINEKAN2(in float input[6], out float output[3]) {
	float output0[bias0_shape0];

	// Layer 0
	for (int j = 0; j < bias0_shape0; j++) {
		output0[j] = bias0[j];
	}

	for (int j = 0; j < amplitudes0_shape0; j++) {
		for (int k = 0; k < amplitudes0_shape1; k++) {
			for (int l = 0; l < amplitudes0_shape2; l++) {
				float s = sin(input[k] * freq0[l] + phase0[k][l]);
				output0[j] += s * amplitudes0[j][k][l];
			}
		}
	}

	// Layer 1
	for (int j = 0; j < bias1_shape0; j++) {
		output[j] = bias1[j];
	}

	for (int j = 0; j < amplitudes1_shape0; j++) {
		for (int k = 0; k < amplitudes1_shape1; k++) {
			for (int l = 0; l < amplitudes1_shape2; l++) {
				float s = sin(output0[k] * freq1[l] + phase1[k][l]);
				output[j] += s * amplitudes1[j][k][l];
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
} */

void BRDFSINEKAN3(in float input[6], out float output[3]) {
	float output0[bias0_shape0];
	float output1[bias1_shape0];

	// Layer 0
	for (int j = 0; j < bias0_shape0; j++) {
		output0[j] = bias0[j];
	}

	for (int j = 0; j < amplitudes0_shape0; j++) {
		for (int k = 0; k < amplitudes0_shape1; k++) {
			for (int l = 0; l < amplitudes0_shape2; l++) {
				float s = sin(input[k] * freq0[l] + phase0[k][l]);
				output0[j] += s * amplitudes0[j][k][l];
			}
		}
	}

	// Layer 1
	for (int j = 0; j < bias1_shape0; j++) {
		output1[j] = bias1[j];
	}

	for (int j = 0; j < amplitudes1_shape0; j++) {
		for (int k = 0; k < amplitudes1_shape1; k++) {
			for (int l = 0; l < amplitudes1_shape2; l++) {
				float s = sin(output0[k] * freq1[l] + phase1[k][l]);
				output1[j] += s * amplitudes1[j][k][l];
			}
		}
	}

	// Layer 2
	for (int j = 0; j < bias2_shape0; j++) {
		output[j] = bias2[j];
	}

	for (int j = 0; j < amplitudes2_shape0; j++) {
		for (int k = 0; k < amplitudes2_shape1; k++) {
			for (int l = 0; l < amplitudes2_shape2; l++) {
				float s = sin(output1[k] * freq2[l] + phase2[k][l]);
				output[j] += s * amplitudes2[j][k][l];
			}
		}
	}

	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
}
