#ifndef INC
#include "BRDFRELUKAN.h"
#else
import INC;
#endif

#ifndef LAYERS
#define LAYERS 3
#endif

void BRDFRELUKAN2(in float input[6], out float output[3]) {
	const float k = 3;

	// Layer 0
	float layer0_output[phase_low0_shape1];

	{
		float activations[phase_low0_shape1] = {};

		const float g = phase_low0_shape1 - k;
		const float r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
			float x_val = input[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				float low        = phase_low0[in_idx][basis_idx];
				float high       = phase_height0[in_idx][basis_idx];
				float x1         = max(0.0, x_val - low);
				float x2         = max(0.0, high - x_val);
				float activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias0_shape0; out_idx++) {
			float sum = equal_size_conv_bias0[out_idx];
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
		float activations[phase_low1_shape1] = {};

		const float g = phase_low1_shape1 - k;
		const float r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
			float x_val = layer0_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				float low        = phase_low1[in_idx][basis_idx];
				float high       = phase_height1[in_idx][basis_idx];
				float x1         = max(0.0, x_val - low);
				float x2         = max(0.0, high - x_val);
				float activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias1_shape0; out_idx++) {
			float sum = equal_size_conv_bias1[out_idx];
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
void BRDFRELUKAN3(in float input[6], out float output[3]) {
	const float k = 3;

	// Layer 0
	float layer0_output[phase_low0_shape1];

	{
		float activations[phase_low0_shape1] = {};

		const float g = phase_low0_shape1 - k;
		const float r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
			float x_val = input[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				float low        = phase_low0[in_idx][basis_idx];
				float high       = phase_height0[in_idx][basis_idx];
				float x1         = max(0.0, x_val - low);
				float x2         = max(0.0, high - x_val);
				float activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias0_shape0; out_idx++) {
			float sum = equal_size_conv_bias0[out_idx];
			for (int basis_idx = 0; basis_idx < phase_low0_shape1; basis_idx++) {
				for (int in_idx = 0; in_idx < equal_size_conv_weight0_shape2; in_idx++) {
					sum += equal_size_conv_weight0[out_idx][basis_idx][in_idx] * activations[basis_idx];
				}
			}
			layer0_output[out_idx] = sum;
		}
	}

	// Layer 1
	float layer1_output[phase_low1_shape1];

	{
		float activations[phase_low1_shape1] = {};

		const float g = phase_low1_shape1 - k;
		const float r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight1_shape2; in_idx++) {
			float x_val = layer0_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low1_shape1; basis_idx++) {
				float low        = phase_low1[in_idx][basis_idx];
				float high       = phase_height1[in_idx][basis_idx];
				float x1         = max(0.0, x_val - low);
				float x2         = max(0.0, high - x_val);
				float activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias1_shape0; out_idx++) {
			float sum = equal_size_conv_bias1[out_idx];
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
		float activations[phase_low2_shape1] = {};

		const float g = phase_low2_shape1 - k;
		const float r = 4.0 * g * g / ((k + 1.0) * (k + 1.0));

		for (int in_idx = 0; in_idx < equal_size_conv_weight2_shape2; in_idx++) {
			float x_val = layer1_output[in_idx];
			for (int basis_idx = 0; basis_idx < phase_low2_shape1; basis_idx++) {
				float low        = phase_low2[in_idx][basis_idx];
				float high       = phase_height2[in_idx][basis_idx];
				float x1         = max(0.0, x_val - low);
				float x2         = max(0.0, high - x_val);
				float activation = x1 * x2 * r;
				activations[basis_idx] += activation * activation;
			}
		}

		for (int out_idx = 0; out_idx < equal_size_conv_bias2_shape0; out_idx++) {
			float sum = equal_size_conv_bias2[out_idx];
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


