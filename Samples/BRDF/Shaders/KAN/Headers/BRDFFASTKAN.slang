#ifndef INC
// #include "BRDFFASTKAN.h"
import FastKAN_6_6_6_3_nCps_6;
// import rgb_decoder_fc;
#else
import INC;
#endif

// #ifndef LAYERS
// #error LAYERS not defined
// #endif

#ifndef LAYERS
#define LAYERS 3
#endif

inline void rbf_forward(in float x, in float grid[], in float denom_inv, out float basis[], int grid_size) {
	for (int i = 0; i < grid_size; i++) {
		float diff = (x - grid[i]) * denom_inv;
		basis[i]   = exp(-diff * diff);
	}
}

inline float spline_linear_forward(in float basis_flat[], int weight_cols /* == basis_size */, in float weights[], int out_idx) {
	float sum = 0.0;
	for (int i = 0; i < weight_cols; i++) {
		sum += basis_flat[i] * weights[out_idx * weight_cols + i];
	}
	return sum;
}

float expm10(float x) {
	typealias T = float;
	return pow(T(1.0) + exp(T(10.0) * x), T(0.1)) - T(1.0);
}

float reluexpm1(float x) {
	return max(0.0, exp(x) - 1.0);
}

float silu(float x) {
	return x / (1.0 + exp(-x));
}

float base_activation(float x) {
	return silu(x);
	// return reluexpm1(x);
	// return expm10(x);
}

void fastkan_layer_forward(
	in float input[], int IN_SIZE,
	out float output[], int OUT_SIZE,
	in float rbf_grid[], int GRID_SIZE,
	in float rbf_denom_inv,
	in float spline_weights[],
	in float base_weights[],
	in float base_bias[]) {

	int FLAT_SIZE = IN_SIZE * GRID_SIZE;

	for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {
		float val = base_bias[out_idx];

		int sw_offset = out_idx * FLAT_SIZE;
		int bw_offset = out_idx * IN_SIZE;

		for (int i = 0; i < IN_SIZE; i++) {
			float x = input[i];

			val += base_activation(x) * base_weights[bw_offset + i];

			int rbf_offset = sw_offset + i * GRID_SIZE;
			for (int j = 0; j < GRID_SIZE; j++) {
				float diff = (x - rbf_grid[j]) * rbf_denom_inv;
				val += exp(-diff * diff) * spline_weights[rbf_offset + j];
			}
		}

		output[out_idx] = val;
	}
}

// #if 1
#if LAYERS == 2
void BRDFFASTKAN2(in float input[6], out float output[3]) {
	float layer0_out[base_weight0_shape0]; // 5

	// Layer 0: [6] -> [5]
	fastkan_layer_forward(
		input, base_weight0_shape1,      // 6
		layer0_out, base_weight0_shape0, // 5
		rbf_grid0, rbf_grid0_shape0,
		rbf_denom_inv0[0], spline_weight0,
		base_weight0, base_bias0);

	// Layer 1: [5] -> [3]
	fastkan_layer_forward(
		layer0_out, base_weight1_shape1, // 5
		output, base_weight1_shape0,     // 5
		rbf_grid1, rbf_grid1_shape0,
		rbf_denom_inv1[0], spline_weight1,
		base_weight1, base_bias1);

	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
}
#endif

#if LAYERS == 3
void BRDFFASTKAN3(in float input[6], out float output[3]) {
	float layer0_out[base_weight0_shape0]; // 5
	float layer1_out[base_weight1_shape0]; // 5

	// Layer 0: [6] -> [5]
	fastkan_layer_forward(
		input, base_weight0_shape1,      // 6
		layer0_out, base_weight0_shape0, // 5
		rbf_grid0, rbf_grid0_shape0,
		rbf_denom_inv0[0], spline_weight0,
		base_weight0, base_bias0);

	// Layer 1: [5] -> [5]
	fastkan_layer_forward(
		layer0_out, base_weight1_shape1, // 5
		layer1_out, base_weight1_shape0, // 5
		rbf_grid1, rbf_grid1_shape0,
		rbf_denom_inv1[0], spline_weight1,
		base_weight1, base_bias1);

	// Layer 2: [5] -> [3]
	fastkan_layer_forward(
		layer1_out, base_weight2_shape1, // 5
		output, base_weight2_shape0,     // 3
		rbf_grid2, rbf_grid2_shape0,
		rbf_denom_inv2[0], spline_weight2,
		base_weight2, base_bias2);

	// Final activation: ReLU(exp(x) - 1)
	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(output[i]) - 1.0);
	}
}
#endif


