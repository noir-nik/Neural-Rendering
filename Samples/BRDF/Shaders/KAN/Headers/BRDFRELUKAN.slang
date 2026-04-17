#ifndef INC
#include "BRDFRELUKAN.h"
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

void BRDFRELUKAN2(in Tp input[6], out Tp output[3]) {
	const Tp k = 3;

	// Layer 0
	Tp layer0_output[phase_low0_shape1];

	{
		Tp activations[phase_low0_shape1] = {};

		const Tp g = phase_low0_shape1 - k;
		const Tp r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
			Tp x_val = input[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				Tp low        = phase_low0[in_idx][basis_idx];
				Tp high       = phase_height0[in_idx][basis_idx];
				Tp x1         = max(0.0, x_val - low);
				Tp x2         = max(0.0, high - x_val);
				Tp activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias0_shape0; out_idx++) {
			Tp sum = equal_size_conv_bias0[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
					sum += equal_size_conv_weight0[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}
			layer0_output[out_idx] = sum;
		}
	}

	// Layer 1
	{
		Tp activations[phase_low1_shape1] = {};

		const Tp g = phase_low1_shape1 - k;
		const Tp r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
			Tp x_val = layer0_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				Tp low        = phase_low1[in_idx][basis_idx];
				Tp high       = phase_height1[in_idx][basis_idx];
				Tp x1         = max(0.0, x_val - low);
				Tp x2         = max(0.0, high - x_val);
				Tp activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias1_shape0; out_idx++) {
			Tp sum = equal_size_conv_bias1[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
					sum += equal_size_conv_weight1[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}
			output[out_idx] = max(0.0, exp(sum) - 1.0);
		}
	}
}

#if LAYERS > 2
void BRDFRELUKAN3(in Tp input[6], out Tp output[3]) {
	const Tp k = 3;

	// Layer 0
	Tp layer0_output[phase_low0_shape1];

	{
		Tp activations[phase_low0_shape1] = {};

		const Tp g = phase_low0_shape1 - k;
		const Tp r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
			Tp x_val = input[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				Tp low        = phase_low0[in_idx][basis_idx];
				Tp high       = phase_height0[in_idx][basis_idx];
				Tp x1         = max(0.0, x_val - low);
				Tp x2         = max(0.0, high - x_val);
				Tp activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias0_shape0; out_idx++) {
			Tp sum = equal_size_conv_bias0[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
					sum += equal_size_conv_weight0[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}
			layer0_output[out_idx] = sum;
		}
	}

	// Layer 1
	Tp layer1_output[phase_low1_shape1];

	{
		Tp activations[phase_low1_shape1] = {};

		const Tp g = phase_low1_shape1 - k;
		const Tp r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
			Tp x_val = layer0_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				Tp low        = phase_low1[in_idx][basis_idx];
				Tp high       = phase_height1[in_idx][basis_idx];
				Tp x1         = max(0.0, x_val - low);
				Tp x2         = max(0.0, high - x_val);
				Tp activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias1_shape0; out_idx++) {
			Tp sum = equal_size_conv_bias1[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
					sum += equal_size_conv_weight1[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}
			layer1_output[out_idx] = sum;
		}
	}

	// Layer 2
	{
		Tp activations[phase_low2_shape1] = {};

		const Tp g = phase_low2_shape1 - k;
		const Tp r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight2_shape2; in_idx++) {
			Tp x_val = layer1_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low2_shape1; basis_idx++) {
				Tp low        = phase_low2[in_idx][basis_idx];
				Tp high       = phase_height2[in_idx][basis_idx];
				Tp x1         = max(0.0, x_val - low);
				Tp x2         = max(0.0, high - x_val);
				Tp activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias2_shape0; out_idx++) {
			Tp sum = equal_size_conv_bias2[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low2_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight2_shape2; in_idx++) {
					sum += equal_size_conv_weight2[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}

			output[out_idx] = max(0.0, exp(sum) - 1.0);
		}
	}
}
#endif


