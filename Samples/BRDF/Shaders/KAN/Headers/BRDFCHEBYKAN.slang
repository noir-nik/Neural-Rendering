#ifndef INC
#include "BRDFCHEBYKAN.h"
// import CHEBYKAN_2_6_12_1170_1209;
// import CHEBYKAN_2_6_8_810_837;
// import CHEBYKAN_2_6_1_180_186;
#else
import INC;
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef _T
#define _T float
#endif

typealias Tp = _T;

// #if LAYERS == 1
// void BRDFCHEBYKAN1(in Tp input[6], out Tp output[3]) {

// }

#if LAYERS == 2
void BRDFCHEBYKAN2(in Tp input[6], out Tp output[3]) {
	// Layer 0: [6] -> [5]
	Tp layer0_out[cheby_coeffs0_shape1];
	for (int o = 0; o < cheby_coeffs0_shape1; o++) {
		Tp sum = Tp(0.0);
		for (int i = 0; i < cheby_coeffs0_shape0; i++) {
			// Normalize to [-1, 1]
			Tp x = tanh(input[i]);

			// T_n(x) = cos(n * acos(x))
			for (int d = 0; d < cheby_coeffs0_shape2; d++) {
				Tp cheby_val = cos(arange0[d] * acos(x));
				sum += cheby_val * cheby_coeffs0[i][o][d];
			}
		}
		layer0_out[o] = sum;
	}

	// Layer 1: [5] -> [3]
	for (int o = 0; o < cheby_coeffs1_shape1; o++) {
		Tp sum = Tp(0.0);
		for (int i = 0; i < cheby_coeffs1_shape0; i++) {
			Tp x = tanh(layer0_out[i]);

			for (int d = 0; d < cheby_coeffs1_shape2; d++) {
				Tp cheby_val = cos(arange1[d] * acos(x));
				sum += cheby_val * cheby_coeffs1[i][o][d];
			}
		}
		output[o] = max(Tp(0.0), exp(sum) - Tp(1.0));
	}
}
#endif

#if LAYERS == 3
void BRDFCHEBYKAN3(in Tp input[6], out Tp output[3]) {
	Tp layer0_out[cheby_coeffs0_shape1];
	for (int o = 0; o < cheby_coeffs0_shape1; o++) {
		Tp sum = Tp(0.0);
		for (int i = 0; i < cheby_coeffs0_shape0; i++) {
			Tp x = tanh(input[i]);

			for (int d = 0; d < cheby_coeffs0_shape2; d++) {
				Tp cheby_val = cos(arange0[d] * acos(x));
				sum += cheby_val * cheby_coeffs0[i][o][d];
			}
		}
		layer0_out[o] = sum;
	}

	// Layer 1: [5] -> [5]
	Tp layer1_out[cheby_coeffs1_shape1];
	for (int o = 0; o < cheby_coeffs1_shape1; o++) {
		Tp sum = Tp(0.0);
		for (int i = 0; i < cheby_coeffs1_shape0; i++) {
			Tp x = tanh(layer0_out[i]);

			for (int d = 0; d < cheby_coeffs1_shape2; d++) {
				Tp cheby_val = cos(arange1[d] * acos(x));
				sum += cheby_val * cheby_coeffs1[i][o][d];
			}
		}
		layer1_out[o] = sum;
	}

	// Layer 2: [5] -> [3]
	for (int o = 0; o < cheby_coeffs2_shape1; o++) {
		Tp sum = Tp(0.0);
		for (int i = 0; i < cheby_coeffs2_shape0; i++) {
			Tp x = tanh(layer1_out[i]);

			for (int d = 0; d < cheby_coeffs2_shape2; d++) {
				Tp cheby_val = cos(arange2[d] * acos(x));
				sum += cheby_val * cheby_coeffs2[i][o][d];
			}
		}
		output[o] = max(Tp(0.0), exp(sum) - Tp(1.0));
	}
}
#endif


