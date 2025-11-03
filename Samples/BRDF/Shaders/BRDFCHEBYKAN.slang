#ifndef INC
#include "BRDFCHEBYKAN.h"
#else
import INC;
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

void BRDFCHEBYKAN2(in float input[6], out float output[3]) {
	// Layer 0: [6] -> [5]
	float layer0_out[cheby_coeffs0_shape1];
	for (int o = 0; o < cheby_coeffs0_shape1; o++) {
		float sum = 0.0;
		for (int i = 0; i < cheby_coeffs0_shape0; i++) {
			// Normalize to [-1, 1]
			float x = tanh(input[i]);

			// T_n(x) = cos(n * acos(x))
			for (int d = 0; d <= arange0_shape0; d++) {
				float cheby_val = cos(arange0[d] * acos(x));
				sum += cheby_val * cheby_coeffs0[i][o][d];
			}
		}
		layer0_out[o] = sum;
	}

	// Layer 1: [5] -> [3]
	for (int o = 0; o < cheby_coeffs1_shape1; o++) {
		float sum = 0.0;
		for (int i = 0; i < cheby_coeffs1_shape0; i++) {
			float x = tanh(layer0_out[i]);

			for (int d = 0; d <= arange1_shape0; d++) {
				float cheby_val = cos(arange2[d] * acos(x));
				sum += cheby_val * cheby_coeffs2[i][o][d];
			}
		}
		output[o] = max(0.0, exp(sum) - 1.0);
	}
}

#if LAYERS > 2
void BRDFCHEBYKAN3(in float input[6], out float output[3]) {
	// Layer 0: [6] -> [5]
	float layer0_out[cheby_coeffs0_shape1];
	for (int o = 0; o < cheby_coeffs0_shape1; o++) {
		float sum = 0.0;
		for (int i = 0; i < cheby_coeffs0_shape0; i++) {
			// Normalize to [-1, 1]
			float x = tanh(input[i]);

			// T_n(x) = cos(n * acos(x))
			for (int d = 0; d <= arange0_shape0; d++) {
				float cheby_val = cos(arange0[d] * acos(x));
				sum += cheby_val * cheby_coeffs0[i][o][d];
			}
		}
		layer0_out[o] = sum;
	}

	// Layer 1: [5] -> [5]
	float layer1_out[cheby_coeffs1_shape1];
	for (int o = 0; o < cheby_coeffs1_shape1; o++) {
		float sum = 0.0;
		for (int i = 0; i < cheby_coeffs1_shape0; i++) {
			float x = tanh(layer0_out[i]);

			for (int d = 0; d <= arange1_shape0; d++) {
				float cheby_val = cos(arange1[d] * acos(x));
				sum += cheby_val * cheby_coeffs1[i][o][d];
			}
		}
		layer1_out[o] = sum;
	}

	// Layer 2: [5] -> [3]
	for (int o = 0; o < cheby_coeffs2_shape1; o++) {
		float sum = 0.0;
		for (int i = 0; i < cheby_coeffs2_shape0; i++) {
			float x = tanh(layer1_out[i]);

			for (int d = 0; d <= arange2_shape0; d++) {
				float cheby_val = cos(arange2[d] * acos(x));
				sum += cheby_val * cheby_coeffs2[i][o][d];
			}
		}
		output[o] = max(0.0, exp(sum) - 1.0);
	}
}
#endif


