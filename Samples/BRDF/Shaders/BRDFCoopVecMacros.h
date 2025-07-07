#define FWD_LAST(idx, in, out, input, output) output = LinearForward<T, out, in>(input, weights, weights_offsets[idx], bias_offsets[idx], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
#define FWD(idx, in, out, input, output) \
	FWD_LAST(idx, in, out, input, output); \
	output = relu(output);