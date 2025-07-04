#define FWD_LAST(in, out, input, output, idx) output = LinearForward<T, out, in>(input, weights, weights_offsets[idx], bias_offsets[idx], CoopVecMatrixLayout::InferencingOptimal, COMPONENT_TYPE);
#define FWD(in, out, input, output, idx) \
	FWD_LAST(in, out, input, output, idx); \
	output = relu(output);