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

static const CoopVecComponentType component_type = CoopVecComponentType::Float32;
static const CoopVecMatrixLayout  matrix_layout  = CoopVecMatrixLayout::InferencingOptimal;

#define rbfidx(input, grid, idx) \
	let                   tmp##idx = (CoopVec<T, NUM_GRIDS>(input[idx]) - grid) * RBF_DENOM_INV; \
	CoopVec<T, NUM_GRIDS> rbf##idx = exp(-(tmp##idx * tmp##idx));

// typedef float T;
// typedef half T;

void BRDFFastKanCoopVec
	//
	<T : __BuiltinFloatingPointType>
	//
	(in T input[6], out T output[3], StructuredBuffer<T> weights) {
	static const T RBF_DENOM_INV = T(3.5);

	CoopVec<T, 6> input0 = CoopVec<T, 6>(input[0], input[1], input[2], input[3], input[4], input[5]);
	CoopVec<T, 5> output0;
	CoopVec<T, 5> output1;
	CoopVec<T, 3> output2;

	constexpr const int layer_dims[4] = int[4](6, 5, 5, 3);

	// ========== LAYER 0: 6 -> 5 ==========
	{
		const int          layer_idx = 0;
		LayerBufferOffsets offs      = pc.offsets[0];

		// Load RBF grid
		CoopVec<T, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis: exp(-((x - grid) * denom_inv)^2)
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);
		rbfidx(input0, grid, 5);

		// Flatten
		CoopVec<T, 48> spline_basis = CoopVec<T, 48>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7],
			rbf5[0], rbf5[1], rbf5[2], rbf5[3], rbf5[4], rbf5[5], rbf5[6], rbf5[7]);

		// Spline linear: 48 -> 5
		CoopVec<T, 5> spline_out;
		spline_out = coopVecMatMul<T, 5, 48>(
			spline_basis, component_type,
			weights, offs.spline_weight_offset, component_type,
			matrix_layout, false, 0);

		// Base
		CoopVec<T, 6> base_in = tanh(input0);
		CoopVec<T, 5> base_out;
		base_out = coopVecMatMulAdd<T, 5, 6>(base_in, component_type,
											 weights, offs.base_weight_offset, component_type,
											 weights, offs.base_bias_offset, component_type,
											 matrix_layout, false, 0);

		// base_out = LinearForward<T, 5, 6>(base_in, weights, offs.base_weight_offset, offs.base_bias_offset, matrix_layout, component_type);

		// Sum
		output0 = spline_out + base_out;
	}

	// ========== LAYER 1: 5 -> 5 ==========
	{
		LayerBufferOffsets offs = pc.offsets[1];

		// Load RBF grid
		CoopVec<T, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);

		// Flatten
		CoopVec<T, 40> spline_basis = CoopVec<T, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 5
		CoopVec<T, 5> spline_out;
		spline_out = coopVecMatMul<T, 5, 40>(spline_basis, component_type,
											 weights, offs.spline_weight_offset, component_type,
											 matrix_layout, false, 0);

		// Base
		CoopVec<T, 5> base_in = tanh(output0);
		CoopVec<T, 5> base_out;
		base_out = coopVecMatMulAdd<T, 5, 5>(
			base_in, component_type,
			weights, offs.base_weight_offset, component_type,
			weights, offs.base_bias_offset, component_type,
			matrix_layout, false, 5 * sizeof(T));

		// Sum
		output1 = spline_out + base_out;
	}

	// ========== LAYER 2: 5 -> 3 ==========
	{
		LayerBufferOffsets offs = pc.offsets[2];

		// Load RBF grid
		CoopVec<T, 8> grid;
		grid.load(weights, offs.rbf_grid_offset);

		// RBF basis
		rbfidx(input0, grid, 0);
		rbfidx(input0, grid, 1);
		rbfidx(input0, grid, 2);
		rbfidx(input0, grid, 3);
		rbfidx(input0, grid, 4);

		// Flatten
		CoopVec<T, 40> spline_basis = CoopVec<T, 40>(
			rbf0[0], rbf0[1], rbf0[2], rbf0[3], rbf0[4], rbf0[5], rbf0[6], rbf0[7],
			rbf1[0], rbf1[1], rbf1[2], rbf1[3], rbf1[4], rbf1[5], rbf1[6], rbf1[7],
			rbf2[0], rbf2[1], rbf2[2], rbf2[3], rbf2[4], rbf2[5], rbf2[6], rbf2[7],
			rbf3[0], rbf3[1], rbf3[2], rbf3[3], rbf3[4], rbf3[5], rbf3[6], rbf3[7],
			rbf4[0], rbf4[1], rbf4[2], rbf4[3], rbf4[4], rbf4[5], rbf4[6], rbf4[7]);

		// Spline linear: 40 -> 3
		CoopVec<T, 3> spline_out;
		spline_out = coopVecMatMul<T, 3, 40>(
			spline_basis, component_type,
			weights, offs.spline_weight_offset, component_type,
			matrix_layout, false, 40 * sizeof(T));

		// Base
		CoopVec<T, 5> base_in = tanh(output1);
		CoopVec<T, 3> base_out;
		base_out = coopVecMatMulAdd<T, 3, 5>(
			base_in, component_type,
			weights, offs.base_weight_offset, component_type,
			weights, offs.base_bias_offset, component_type,
			matrix_layout, false, 5 * sizeof(T));

		// base_out = LinearForward<T, 3, 5>(base_in, weights, offs.base_weight_offset, offs.base_bias_offset, matrix_layout, component_type);

		// Sum
		output2 = spline_out + base_out;
	}

	// max exp
	output2 = max(CoopVec<T, 3>(T(0.0)), exp(output2) - CoopVec<T, 3>(T(1.0)));

	// Final output
	output[0] = output2[0];
	output[1] = output2[1];
	output[2] = output2[2];
}


