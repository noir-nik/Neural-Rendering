module BRDFFASTKAN;

import BRDFFASTKAN_v0;
import BRDFFASTKAN_v1;
import BRDFFASTKAN_v2;
import BRDFFASTKAN_v3;

#ifndef INC
#include "BRDFFASTKAN_2_5_nCps8.slang"
#else
import INC;
#endif

#define PPCAT_NX(A, B) A##B
#define PPCAT(A, B) PPCAT_NX(A, B)

#ifndef _T
#define _T float
#endif

internal typealias Tp = _T;

#ifndef LAYERS
#define LAYERS 3
#endif

#ifndef FASTKAN_VERSION
#define FASTKAN_VERSION 2
#endif

#ifndef fastkan_layer_forward

#if FASTKAN_VERSION == 0
#define fastkan_layer_forward( \
	input, IN_SIZE, \
	output, OUT_SIZE, \
	rbf_grid, GRID_SIZE, \
	rbf_denom_inv, \
	spline_weights, \
	base_weights, \
	base_bias) \
	fastkan_layer_forward_v0<Tp, IN_SIZE, GRID_SIZE>( \
		input, IN_SIZE, \
		output, OUT_SIZE, \
		rbf_grid, GRID_SIZE, \
		rbf_denom_inv, \
		spline_weights, \
		base_weights, \
		base_bias)

#else // FASTKAN_VERSION == 0
#define fastkan_layer_forward PPCAT(fastkan_layer_forward_v, FASTKAN_VERSION)
#endif // FASTKAN_VERSION == 0

#endif // fastkan_layer_forward

// #if 1
#if LAYERS == 2
public void BRDFFASTKAN2(in Tp input[6], out Tp output[3]) {
	Tp layer0_out[base_weight0_shape0]; // 5

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
public void BRDFFASTKAN3(in Tp input[6], out Tp output[3]) {
	Tp layer0_out[base_weight0_shape0]; // 5
	Tp layer1_out[base_weight1_shape0]; // 5

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


