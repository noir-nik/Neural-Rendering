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

typealias FTy = float;

static const FTy RBF_DENOM_INV = 3.5;

#define rbfidx(input, grid, idx) \
	let                     tmp##idx = (CoopVec<FTy, NUM_GRIDS>(input[idx]) - grid) * RBF_DENOM_INV; \
	CoopVec<FTy, NUM_GRIDS> rbf##idx = exp(-(tmp##idx * tmp##idx));

void BRDFFastKanCoopVec(in FTy input[6], out FTy output[3], StructuredBuffer<FTy> weights) {
	CoopVec<FTy, 6> input0 = CoopVec<FTy, 6>(input[0], input[1], input[2], input[3], input[4], input[5]);
	CoopVec<FTy, 5> output0;
	CoopVec<FTy, 5> output1;
	CoopVec<FTy, 3> output2;

	const int layer_dims[4] = int[4](6, 5, 5, 3);

	// ========== LAYER 0: 6 -> 5 ==========
	{
		const int          layer_idx = 0;
		LayerBufferOffsets offs      = pc.offsets[0];

		// Load RBF grid
		CoopVec<FTy, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 6 inputs: exp(-((x - grid) * denom_inv)^2)
		// Each input -> 8 basis values, total 48 values
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);
		rbfidx(input0, grid, 5);

		// Flatten
		CoopVec<FTy, 48> spline_basis = CoopVec<FTy, 48>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7],
			rbf5[0], rbf5[1], rbf5[2], rbf5[3], rbf5[4], rbf5[5], rbf5[6], rbf5[7]);

		// Spline linear: 48 -> 5
		CoopVec<FTy, 5> spline_out;
		spline_out = coopVecMatMul<FTy, 5, 48>(
			spline_basis, CoopVecComponentType::Float32,
			weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<FTy, 6> base_in = tanh(input0);
		CoopVec<FTy, 5> base_out;
		base_out = coopVecMatMulAdd<FTy, 5, 6>(base_in, CoopVecComponentType::Float32,
											   weights, offs.base_weight_offset, CoopVecComponentType::Float32,
											   weights, offs.base_bias_offset, CoopVecComponentType::Float32,
											   CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// base_out = LinearForward<FTy, 5, 6>(base_in, weights, offs.base_weight_offset, offs.base_bias_offset, CoopVecMatrixLayout::InferencingOptimal, CoopVecComponentType::Float32);

		// Sum
		output0 = spline_out + base_out;
	}

	// ========== LAYER 1: 5 -> 5 ==========
	{
		LayerBufferOffsets offs = pc.offsets[1];

		// Load RBF grid
		CoopVec<FTy, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 5 inputs
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);

		// Flatten
		CoopVec<FTy, 40> spline_basis = CoopVec<FTy, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 5
		CoopVec<FTy, 5> spline_out;
		spline_out = coopVecMatMul<FTy, 5, 40>(spline_basis, CoopVecComponentType::Float32,
											   weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
											   CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<FTy, 5> base_in = tanh(output0);
		CoopVec<FTy, 5> base_out;
		base_out = coopVecMatMulAdd<FTy, 5, 5>(
			base_in, CoopVecComponentType::Float32,
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
		CoopVec<FTy, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis for 5 inputs
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);

		// Flatten
		CoopVec<FTy, 40> spline_basis = CoopVec<FTy, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 3
		CoopVec<FTy, 3> spline_out;
		spline_out = coopVecMatMul<FTy, 3, 40>(
			spline_basis, CoopVecComponentType::Float32,
			weights, offs.spline_weight_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Base
		CoopVec<FTy, 5> base_in = tanh(output1);
		CoopVec<FTy, 3> base_out;
		base_out = coopVecMatMulAdd<FTy, 3, 5>(
			base_in, CoopVecComponentType::Float32,
			weights, offs.base_weight_offset, CoopVecComponentType::Float32,
			weights, offs.base_bias_offset, CoopVecComponentType::Float32,
			CoopVecMatrixLayout::InferencingOptimal, false, 0);

		// Sum
		output2 = spline_out + base_out;
	}

	
	// max exp
	output2 = max(CoopVec<FTy, 3>(0.0f), exp(output2) - CoopVec<FTy, 3>(1.0f));

	// Final output
	output[0] = output2[0];
	output[1] = output2[1];
	output[2] = output2[2];
}


