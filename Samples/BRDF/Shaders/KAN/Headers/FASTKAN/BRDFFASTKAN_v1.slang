import BRDFFASTKAN_shared;

// out_idx loop is outside
void fastkan_layer_forward_v1<T : __BuiltinFloatingPointType>(
	in T input[], int IN_SIZE,
	out T output[], int OUT_SIZE,
	in T rbf_grid[], int GRID_SIZE,
	in T rbf_denom_inv,
	in T spline_weights[],
	in T base_weights[],
	in T base_bias[]) {

	int FLAT_SIZE = IN_SIZE * GRID_SIZE;

	for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {
		T val = base_bias[out_idx];

		int sw_offset = out_idx * FLAT_SIZE;
		int bw_offset = out_idx * IN_SIZE;

		for (int i = 0; i < IN_SIZE; i++) {
			T x = input[i];

			val += fastkan_base_activation(x) * base_weights[bw_offset + i];

			int rbf_offset = sw_offset + i * GRID_SIZE;
			for (int j = 0; j < GRID_SIZE; j++) {
				T diff = (x - rbf_grid[j]) * rbf_denom_inv;
				val += exp(-diff * diff) * spline_weights[rbf_offset + j];
			}
		}

		output[out_idx] = val;
	}
}

