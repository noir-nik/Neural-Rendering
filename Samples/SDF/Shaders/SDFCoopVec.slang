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
