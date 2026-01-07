import CoopVec;

struct LayerBufferOffsetsRelu {
	uint phase_low_offset;
	uint phase_height_offset;
	uint equal_size_conv_weight_offset;
	uint equal_size_conv_bias_offset;
};

#define relu_input_idx(idx) \
	CoopVec<T, 11> phase_low_##idx, phase_height_##idx; \
	phase_low_##idx.load(weights, offs.phase_low_offset + idx * num_basis * sizeof(T)); \
	phase_height_##idx.load(weights, offs.phase_height_offset + idx * num_basis * sizeof(T)); \
	CoopVec<T, 11> input_elem_vec##idx = CoopVec<T, 11>(x_input[idx]); \
	CoopVec<T, 11> x1_##idx            = max(CoopVec<T, 11>(0.0), input_elem_vec##idx - phase_low_##idx); \
	CoopVec<T, 11> x2_##idx            = max(CoopVec<T, 11>(0.0), phase_height_##idx - input_elem_vec##idx); \
	CoopVec<T, 11> act_##idx           = x1_##idx * x2_##idx * r; \
	act_##idx                          = act_##idx * act_##idx

void BRDFReluKanCoopVec<T : __BuiltinFloatingPointType>(in T input[6], out T output[3], StructuredBuffer<T> weights, LayerBufferOffsetsRelu offs) {
	const T   k          = T(3.0);
	const T   g          = T(8.0); // grid_size - k = 11 - 3
	const T   r          = T(4.0) * g * g / ((k + T(1.0)) * (k + T(1.0)));
	const int num_basis  = 11; // g + k
	const int input_dim  = 6;
	const int output_dim = 3;

	CoopVec<T, 6> x_input = CoopVec<T, 6>(
		input[0], input[1], input[2], input[3], input[4], input[5]);

	// basis activations for each input dimension
	// Result is [6, 11] flatten to [66]

	relu_input_idx(0);
	relu_input_idx(1);
	relu_input_idx(2);
	relu_input_idx(3);
	relu_input_idx(4);
	relu_input_idx(5);

	// Flatten [input_dim=6, num_basis=11]
	CoopVec<T, 66> activation_map = CoopVec<T, 66>(
		act_0[0], act_0[1], act_0[2], act_0[3], act_0[4], act_0[5], act_0[6], act_0[7], act_0[8], act_0[9], act_0[10],
		act_1[0], act_1[1], act_1[2], act_1[3], act_1[4], act_1[5], act_1[6], act_1[7], act_1[8], act_1[9], act_1[10],
		act_2[0], act_2[1], act_2[2], act_2[3], act_2[4], act_2[5], act_2[6], act_2[7], act_2[8], act_2[9], act_2[10],
		act_3[0], act_3[1], act_3[2], act_3[3], act_3[4], act_3[5], act_3[6], act_3[7], act_3[8], act_3[9], act_3[10],
		act_4[0], act_4[1], act_4[2], act_4[3], act_4[4], act_4[5], act_4[6], act_4[7], act_4[8], act_4[9], act_4[10],
		act_5[0], act_5[1], act_5[2], act_5[3], act_5[4], act_5[5], act_5[6], act_5[7], act_5[8], act_5[9], act_5[10]);

	// Conv2d as matrix multiply
	// Conv2d(1, output_size=3, kernel=(11, 6)) with input shape [1, 11, 6]
	// Weight shape: [3, 1, 11, 6] -> flattened to [3, 66]
	// output = activation_map @ weight.T + bias
	CoopVec<T, 3> result;
	result = coopVecMatMulAdd<T, 3, 66>(
		activation_map,
		CoopVecComponentType::Float32,
		weights,
		offs.equal_size_conv_weight_offset,
		CoopVecComponentType::Float32,
		weights,
		offs.equal_size_conv_bias_offset,
		CoopVecComponentType::Float32,
		CoopVecMatrixLayout::InferencingOptimal,
		false,
		66 * sizeof(T) // stride in bytes
	);

	// max exp
	result = max(CoopVec<T, 3>(0.0f), exp(result) - CoopVec<T, 3>(1.0f));

	output[0] = result[0];
	output[1] = result[1];
	output[2] = result[2];
}

#undef relu_input_idx


