#include "BRDFConfig.h"
#include "BRDFCoopVecMacros.h"

import CoopVec;

void BRDFCoopVec<T : __BuiltinFloatingPointType>(in T input[6], out T output[3], StructuredBuffer<T> weights, uint32_t weights_offsets[5], uint32_t bias_offsets[5]) {

	CoopVec<T, 6>  input0;
	CoopVec<T, 64> output0;
	CoopVec<T, 64> output1;
	CoopVec<T, 64> output2;
	CoopVec<T, 6>  output3;
	CoopVec<T, 3>  output4;

	input0 = { input[0], input[1], input[2], input[3], input[4], input[5] };

	// FWD(idx, in_size, out_size, input, output)
	FWD(0, 6, 64, input0, output0);
	FWD(1, 64, 64, output0, output1);
	FWD(2, 64, 64, output1, output2);
	FWD(3, 64, 6, output2, output3);
	FWD_LAST(4, 6, 3, output3, output4);

	output[0] = output4[0];
	output[1] = output4[1];
	output[2] = output4[2];
}


