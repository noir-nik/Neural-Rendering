#include "BRDFConfig.h"
import CoopVec;

func BRDFCoopVec<T : __BuiltinFloatingPointType>(
	vector<T, 3>        view,
	vector<T, 3>        light,
	StructuredBuffer<T> weights,
	uint32_t            weights_offsets[5],
	uint32_t            bias_offsets[5])
	->vector<T, 3> {

	CoopVec<T, 6>  input;
	CoopVec<T, 64> output0;
	CoopVec<T, 64> output1;
	CoopVec<T, 64> output2;
	CoopVec<T, 6>  output3;
	CoopVec<T, 3>  output4;

	// Test
	// printf("output4 = %f, %f, %f\n", weights[bias_offsets[4] / sizeof(T) + 0], weights[bias_offsets[4] / sizeof(T) + 1], weights[bias_offsets[4] / sizeof(T) + 2]);

	input   = { view.x, view.y, view.z, light.x, light.y, light.z };
	output0 = LinearForward<T, 64, 6>(input, weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output0 = relu(output0);
	output1 = LinearForward<T, 64, 64>(output0, weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output1 = relu(output1);
	output2 = LinearForward<T, 64, 64>(output1, weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output2 = sin(output2);
	output3 = LinearForward<T, 6, 64>(output2, weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
	output3 = relu(output3);
	output4 = LinearForward<T, 3, 6>(output3, weights, weights_offsets[4], bias_offsets[4], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);

	return { output4[0], output4[1], output4[2] };
}


