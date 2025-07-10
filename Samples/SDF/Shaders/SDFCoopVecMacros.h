#pragma once
#define FWD_LAST(idx, in, out, input, output) output = LinearForward<T, out, in>(input, weights, weights_offsets[idx], bias_offsets[idx], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
#define FWD(idx, in, out, input, output) \
	FWD_LAST(idx, in, out, input, output); \
	output = sin(CoopVec<T, out>(T(30.0)) * output);

// output = sin(output);

// #define FWD_SINE(idx, in, out, input, output) \
// 	FWD_LAST(idx, in, out, input, output); \
// 	output = sin(output);