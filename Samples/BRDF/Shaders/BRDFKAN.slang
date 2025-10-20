// #include "BRDFConfig.h"

#include "BRDFKAN-alum-bronze_6_5_5_3.h"
// #include "BRDFKAN.h"

// #include "BRDFKAN_0.006604_flat.h"

void rbf_forward(in float x, in float grid[], in float denom_inv, out float basis[], int grid_size) {
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

float base_activation(float x) {
	return tanh(x);
}

float silu(float x) {
	return x / (1.0 + exp(-x));
}

void fastkan_layer_forward(
	in float input[], int input_dim,    // 6 -- first layer
	out float output[], int output_dim, // 5 -- first layer
	in float rbf_grid[], in float rbf_denom_inv,
	in float spline_weights[],
	in float base_weights[],
	in float base_bias[],
	int      spline_weight_cols, // 48 todo: make a template parameter
	int      grid_size) {
	// RBF basis
	float basis_flat[48]; // Max: 6 inputs * 8 grids = 48
	int   basis_idx = 0;  // 0 ... 48 (input_dim * grid_size)

	// input (6) -> basis_flat (48)
	// input[0] -> basis_flat[0, 1, 2, 3, 4, 5, 6, 7]
	// input[1] -> basis_flat[8, 9, 10, 11, 12, 13, 14, 15]
	for (int i = 0; i < input_dim; i++) {
		// constexpr const int grid_size = sizeof(rbf_grid) / sizeof(rbf_grid[0]);
		float basis[8]; // grid_size = 8
		rbf_forward(input[i], rbf_grid, rbf_denom_inv, basis, grid_size);
		for (int j = 0; j < grid_size; j++) {
			basis_flat[basis_idx++] = basis[j];
		}
	}

	// Spline and base
	// 0 ... 5 -- first layer
	for (int out_idx = 0; out_idx < output_dim; out_idx++) {

		// Spline linear (no bias)
		float spline_val = 0.0;
		
		// 0 ... 48 -- first layer
		for (int j = 0; j < basis_idx; j++) {
			spline_val += basis_flat[j] * spline_weights[out_idx * basis_idx /* == 48 == spline_weight_cols */ + j];
		}

		// spline_val = spline_linear_forward(basis_flat, basis_idx, spline_weights, out_idx);

		// Base linear + bias
		float base_val = base_bias[out_idx];
		for (int i = 0; i < input_dim; i++) {
			base_val += base_activation(input[i]) * base_weights[out_idx * input_dim + i];
		}

		output[out_idx] = spline_val + base_val;
	}
}

void BRDFKAN(in float input[6], out float output[3]) {
	float layer0_out[5];
	float layer1_out[5];
	float layer2_out[3];

	// Layer 0: [6] -> [5]
	fastkan_layer_forward(
		input, base_weight0_shape1,      // 6
		layer0_out, base_weight0_shape0, // 5
		rbf_grid0, rbf_denom_inv0[0],
		spline_weight0, base_weight0, base_bias0,
		spline_weight0_shape1, // 6 * 8 = 48,
		rbf_grid0_shape0);

	// Layer 1: [5] -> [5]
	fastkan_layer_forward(
		layer0_out, base_weight1_shape1, // 5
		layer1_out, base_weight1_shape0, // 5
		rbf_grid1, rbf_denom_inv1[0],
		spline_weight1, base_weight1, base_bias1,
		spline_weight1_shape1, // 5 * 8 = 40
		rbf_grid1_shape0);

	// Layer 2: [5] -> [3]
	fastkan_layer_forward(
		layer1_out, base_weight2_shape1, // 5
		layer2_out, base_weight2_shape0, // 3
		rbf_grid2, rbf_denom_inv2[0],
		spline_weight2, base_weight2, base_bias2,
		spline_weight2_shape1, // 5 * 8 = 40
		rbf_grid2_shape0);

	// Final activation: ReLU(exp(x) - 1)
	for (int i = 0; i < 3; i++) {
		output[i] = max(0.0, exp(layer2_out[i]) - 1.0);
	}
}


