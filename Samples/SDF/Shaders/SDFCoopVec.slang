#include "SDFConfig.h"
import CoopVec;
// clang-format off

func SDFCoopVec<T : __BuiltinFloatingPointType>(vector<T, 3> p, StructuredBuffer<T> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4])->T {
	if (length(p) > T(1.)) {
		return length(p) - T(.8);
	}
	CoopVec<T, 3>  input;
	CoopVec<T, 16> output0;
	CoopVec<T, 16> output1;
	CoopVec<T, 16> output2;
	CoopVec<T, 1>  output3;

	input   = { p.x, p.y, p.z };
	output0 = ng::LinearForward<T, 16, 3>(input, sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output0 = sin(output0);
	output1 = ng::LinearForward<T, 16, 16>(output0, sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output1 = sin(output1) + output0;
	output2 = ng::LinearForward<T, 16, 16>(output1, sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output2 = sin(output2) / CoopVec<T, 16>(T(1.4)) + output1;
	output3 = ng::LinearForward<T, 1, 16>(output2, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);

	return output3[0];
}

func D_SDFCoopVec<T : __BuiltinFloatingPointType>(vector<T, 3> p, StructuredBuffer<T> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4])->vector<T, 3> {
	if (length(p) > T(1.)) {
		return normalize(p);
	}

	CoopVec<T, 3>  input;
	CoopVec<T, 16> z0;
	CoopVec<T, 16> z1;
	CoopVec<T, 16> z2;
	CoopVec<T, 1>  z3;

	CoopVec<T, 16> output0;
	CoopVec<T, 16> output1;
	CoopVec<T, 16> output2;
	CoopVec<T, 1>  output3;

	input   = { p.x, p.y, p.z };
	z0      = ng::LinearForward<T, 16, 3>(input, sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output0 = sin(z0);
	z1      = ng::LinearForward<T, 16, 16>(output0, sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output1 = sin(z1) + output0;
	z2      = ng::LinearForward<T, 16, 16>(output1, sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output2 = sin(z2) / CoopVec<T, 16>(T(1.4)) + output1;
	z3      = ng::LinearForward<T, 1, 16>(output2, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);

	CoopVec<T, 1>  d_output3 = { T(1.0) };
	CoopVec<T, 16> d_output2;
	CoopVec<T, 16> d_output1;
	CoopVec<T, 16> d_output0;
	CoopVec<T, 3> d_input = { T(0.0), T(0.0), T(0.0) };

	// CoopVec<T, 16> d_output2 = ng::LinearForward<T, 16, 1>(d_output3, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// CoopVec<T, 16> d_output1 = d_output2;
	// d_output1 = d_output1 + ng::LinearForward<T, 16, 16>(d_output2 * cos(z2) / CoopVec<T, 16>(T(1.4)), sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// CoopVec<T, 16> d_output0 =  d_output1;
	// d_output0 = d_output0 + ng::LinearForward<T, 16, 16>(d_output1 * cos(z1) , sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_input = ng::LinearForward<T, 3, 16>(d_output0 * cos(z0), sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);

	// d_output2 = ng::LinearForward<T, 16, 1>(d_output3, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output1 = ng::LinearForward<T, 16, 16>(d_output2 * cos(z2) / CoopVec<T, 16>(T(1.4)), sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output1 = d_output1 + d_output2;
	// d_output0 = ng::LinearForward<T, 16, 16>(d_output1 * cos(z1) , sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output0 = d_output0+ d_output1;
	// d_input = ng::LinearForward<T, 3, 16>(d_output0 * cos(z0), sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);

	// d_output2 = ng::LinearForward<T, 16, 1>(d_output3, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output2 = d_output2 * cos(z2) / CoopVec<T, 16>(T(1.4));
	// d_output1 = ng::LinearForward<T, 16, 16>(d_output2, sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output1 = d_output1 * cos(z1) + d_output2;
	// d_output0 = ng::LinearForward<T, 16, 16>(d_output1, sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);
	// d_output0 = d_output0 * cos(z0) + d_output1;
	// d_input = ng::LinearForward<T, 3, 16>(d_output0, sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE, true);

	// for (int i = 0; i < 16; ++i) {
	// 	d_output2[i] = d_output3[0] * sdf_weights[weights_offsets[3] / sizeof(T) + i];
	// }

	// // Skip connection
	// for (int i = 0; i < 16; ++i) {
	// 	d_output1[i] = d_output2[i];
	// }
	
	// for (int j = 0; j < 16; ++j) {
	// 	for (int i = 0; i < 16; ++i) {
	// 		d_output1[j] += d_output2[i] * sdf_weights[weights_offsets[2] / sizeof(T) + i * 16 + j] * cos(z2[i]) / T(1.4);
	// 	}
	// }

	// // Skip connection
	// for (int i = 0; i < 16; ++i) {
	// 	d_output0[i] = d_output1[i];
	// }

	// for (int j = 0; j < 16; ++j) {
	// 	for (int i = 0; i < 16; ++i) {
	// 		d_output0[j] += d_output1[i] * sdf_weights[weights_offsets[1] / sizeof(T) + i * 16 + j] * cos(z1[i]);
	// 	}
	// }

	// for (int j = 0; j < 3; ++j) {
	// 	for (int i = 0; i < 16; ++i) {
	// 		d_input[j] += d_output0[i] * sdf_weights[weights_offsets[0] / sizeof(T) + i * 3 + j] * cos(z0[i]);
	// 	}
	// }

	return {d_input[0], d_input[1], d_input[2]};
}
