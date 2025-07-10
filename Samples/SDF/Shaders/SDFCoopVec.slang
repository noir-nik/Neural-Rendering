#include "SDFConfig.h"
#include "SDFCoopVecMacros.h"
import CoopVec;
// clang-format off

void SDFCoopVec<T : __BuiltinFloatingPointType>(in T input[3], out T output[1], StructuredBuffer<T> weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4]) {
	vector<T, 3> p = { input[0], input[1], input[2] };
	if (length(p) > T(1.)) {
		output[0] = length(p) - T(.8);
		return;
	}
	CoopVec<T, 3>  input0;
	CoopVec<T, 16> output0;
	CoopVec<T, 16> output1;
	CoopVec<T, 16> output2;
	CoopVec<T, 1>  output3;

	input0 = { input[0], input[1], input[2] };
	FWD(0, 3, 16, input0, output0);
	FWD(1, 16, 16, output0, output1);
	output1 = output1 + output0;
	FWD(2, 16, 16, output1, output2);
	output2 = (output2) / CoopVec<T, 16>(T(1.4)) + output1;
	FWD_LAST(3, 16, 1, output2, output3);

	output[0] = output3[0];
}

