#define MAX_LAYERS 4
#define NUM_GRIDS 8

import CoopVec;

struct LayerBufferOffsets {
	uint rbf_grid_offset;
	uint spline_weight_offset;
	uint base_weight_offset;
	uint base_bias_offset;
};

struct PushConstants {
	int                num_layers;
	LayerBufferOffsets offsets[MAX_LAYERS];
};

[[vk::push_constant]]
PushConstants pc;

[[vk::binding(0, 0)]]
StructuredBuffer<float> weights;


static const float RBF_DENOM_INV = 3.5;

#define rbffunc(input, grid, idx) \
	let                       tmp##idx = (CoopVec<float, NUM_GRIDS>(input[idx]) - grid) * RBF_DENOM_INV; \
	CoopVec<float, NUM_GRIDS> rbf##idx = exp(-(tmp##idx * tmp##idx));


void BRDFFASTKAN(in float input[6], out float output[3]) {
	CoopVec<float, 6> input0 = CoopVec<float, 6>(input[0], input[1], input[2], input[3], input[4], input[5]);
	CoopVec<float, 5> output0;
	CoopVec<float, 5> output1;
	CoopVec<float, 3> output2;

	// ========== LAYER 0: 6 -> 5 ==========
	{
		LayerBufferOffsets offs = pc.offsets[0];

		// Load RBF grid
		CoopVec<float, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 6 inputs: exp(-((x - grid) * denom_inv)^2)
		// Each input -> 8 basis values, total 48 values
		// rbffunc(input0, grid, 0);
		let               tmp0 = (CoopVec<float, NUM_GRIDS>(input0[0]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf0 = exp(-(tmp0 * tmp0));
		let               tmp1 = (CoopVec<float, NUM_GRIDS>(input0[1]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf1 = exp(-(tmp1 * tmp1));
		let               tmp2 = (CoopVec<float, NUM_GRIDS>(input0[2]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf2 = exp(-(tmp2 * tmp2));
		let               tmp3 = (CoopVec<float, NUM_GRIDS>(input0[3]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf3 = exp(-(tmp3 * tmp3));
		let               tmp4 = (CoopVec<float, NUM_GRIDS>(input0[4]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf4 = exp(-(tmp4 * tmp4));
		let               tmp5 = (CoopVec<float, NUM_GRIDS>(input0[5]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf5 = exp(-(tmp5 * tmp5));

		// Flatten
		CoopVec<float, 48> spline_basis = CoopVec<float, 48>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7],
			rbf5[0], rbf5[1], rbf5[2], rbf5[3], rbf5[4], rbf5[5], rbf5[6], rbf5[7]);

		// Spline linear: 48 -> 5
		CoopVec<float, 5> spline_out;
		spline_out = coopVecMatMul<float, 5, 48>(
			spline_basis, CoopVecComponentType::Float32,
			weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<float, 6> silu_input = input0 / (CoopVec<float, 6>(1.0) + exp(-input0));
		CoopVec<float, 5> base_out;
		base_out = coopVecMatMulAdd<float, 5, 6>(silu_input, CoopVecComponentType::Float32,
												 weights, offs.base_weight_offset, CoopVecComponentType::Float32,
												 weights, offs.base_bias_offset, CoopVecComponentType::Float32,
												 CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// base_out = LinearForward<float, 5, 6>(silu_input, weights, offs.base_weight_offset, offs.base_bias_offset, CoopVecMatrixLayout::InferencingOptimal, CoopVecComponentType::Float32);

		// Sum
		output0 = spline_out + base_out;
	}

	// ========== LAYER 1: 5 -> 5 ==========
	{
		LayerBufferOffsets offs = pc.offsets[1];

		// Load RBF grid
		CoopVec<float, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 5 inputs
		let               tmp0 = (CoopVec<float, NUM_GRIDS>(output0[0]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf0 = exp(-(tmp0 * tmp0));
		let               tmp1 = (CoopVec<float, NUM_GRIDS>(output0[1]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf1 = exp(-(tmp1 * tmp1));
		let               tmp2 = (CoopVec<float, NUM_GRIDS>(output0[2]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf2 = exp(-(tmp2 * tmp2));
		let               tmp3 = (CoopVec<float, NUM_GRIDS>(output0[3]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf3 = exp(-(tmp3 * tmp3));
		let               tmp4 = (CoopVec<float, NUM_GRIDS>(output0[4]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf4 = exp(-(tmp4 * tmp4));

		// Flatten
		CoopVec<float, 40> spline_basis = CoopVec<float, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 5
		CoopVec<float, 5> spline_out;
		spline_out = coopVecMatMul<float, 5, 40>(spline_basis, CoopVecComponentType::Float32,
												 weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
												 CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<float, 5> silu_input = output0 / (CoopVec<float, 5>(1.0) + exp(-output0));
		CoopVec<float, 5> base_out;
		base_out = coopVecMatMulAdd<float, 5, 5>(
			silu_input, CoopVecComponentType::Float32,
			weights, offs.base_weight_offset, CoopVecComponentType::Float32,
			weights, offs.base_bias_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Sum
		output1 = spline_out + base_out;
	}

	// ========== LAYER 2: 5 -> 3 ==========
	{
		LayerBufferOffsets offs = pc.offsets[2];

		// Load RBF grid
		CoopVec<float, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 5 inputs
		let               tmp0 = (CoopVec<float, NUM_GRIDS>(output1[0]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf0 = exp(-(tmp0 * tmp0));
		let               tmp1 = (CoopVec<float, NUM_GRIDS>(output1[1]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf1 = exp(-(tmp1 * tmp1));
		let               tmp2 = (CoopVec<float, NUM_GRIDS>(output1[2]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf2 = exp(-(tmp2 * tmp2));
		let               tmp3 = (CoopVec<float, NUM_GRIDS>(output1[3]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf3 = exp(-(tmp3 * tmp3));
		let               tmp4 = (CoopVec<float, NUM_GRIDS>(output1[4]) - grid) * RBF_DENOM_INV;
		CoopVec<float, 8> rbf4 = exp(-(tmp4 * tmp4));

		// Flatten
		CoopVec<float, 40> spline_basis = CoopVec<float, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 3
		CoopVec<float, 3> spline_out;
		spline_out = coopVecMatMul<float, 3, 40>(
			spline_basis, CoopVecComponentType::Float32,
			weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<float, 5> silu_input = output1 / (CoopVec<float, 5>(1.0) + exp(-output1));
		CoopVec<float, 3> base_out;
		base_out = coopVecMatMulAdd<float, 3, 5>(
			silu_input, CoopVecComponentType::Float32,
			weights, offs.base_weight_offset, CoopVecComponentType::Float32,
			weights, offs.base_bias_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Sum
		output2 = spline_out + base_out;
	}

	// Final output
	output[0] = output2[0];
	output[1] = output2[1];
	output[2] = output2[2];
}
