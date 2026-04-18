#pragma once
#define FWD(idx, in, out, input, output) output = LinearForward<T, out, in>(input, weights, weights_offsets[idx], bias_offsets[idx], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
#define FWD_RELU(idx, in, out, input, output) \
	FWD(idx, in, out, input, output); \
	output = relu(output);

#define FWD_SINE(idx, in, out, input, output) \
	FWD(idx, in, out, input, output); \
	output = sin(output);
	
// #define FWD_RELUEXPM1(idx, in, out, input, output) \
// 	FWD(idx, in, out, input, output); \
// 	output = relu(exp(output) - T(1.0));
	// output = sin(output);
	// max(T(0.0), exp(output[i] - T(1.0)));