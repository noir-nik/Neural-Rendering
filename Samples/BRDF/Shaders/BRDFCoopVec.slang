#include "BRDFConfig.h"
#include "BRDFCoopVecMacros.h"

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

	input = { view.x, view.y, view.z, light.x, light.y, light.z };

	FWD(6, 64, input, output0, 0);
	FWD(64, 64, output0, output1, 1);
	FWD(64, 64, output1, output2, 2);
	FWD(64, 6, output2, output3, 3);
	FWD(6, 3, output3, output4, 4);

	return { output4[0], output4[1], output4[2] };
}


