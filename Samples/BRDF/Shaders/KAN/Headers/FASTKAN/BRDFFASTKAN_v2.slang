import BRDFFASTKAN_shared;

// out_idx loop is inside
void fastkan_layer_forward_v2<T : __BuiltinFloatingPointType>(
	in T input[], int IN_SIZE,
	out T output[], int OUT_SIZE,
	in T rbf_grid[], int GRID_SIZE,
	in T rbf_denom_inv,
	in T spline_weights[],
	in T base_weights[],
	in T base_bias[]) {

	int FLAT_SIZE = IN_SIZE * GRID_SIZE;

	for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {
		output[out_idx] = base_bias[out_idx];
	}

	for (int i = 0; i < IN_SIZE; i++) {
		T x  = input[i];
		T tx = fastkan_base_activation(x);

		// Base
		for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {
			output[out_idx] += tx * base_weights[out_idx * IN_SIZE + i];
		}

		// Spline
		for (int j = 0; j < GRID_SIZE; j++) {
			T   diff      = (x - rbf_grid[j]) * rbf_denom_inv;
			T   rbf       = exp(-diff * diff);
			int sw_offset = i * GRID_SIZE + j;

			for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {
				output[out_idx] += rbf * spline_weights[out_idx * FLAT_SIZE + sw_offset];
			}
		}
	}
}


