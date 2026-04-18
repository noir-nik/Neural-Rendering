import BRDFFASTKAN_shared;

public void fastkan_layer_forward_v0<T : __BuiltinFloatingPointType, int IN, int GRID>(
	in T input[], constexpr int IN_SIZE,
	out T output[], constexpr int OUT_SIZE,
	in T rbf_grid[], constexpr int GRID_SIZE,
	in T rbf_denom_inv,
	in T spline_weights[],
	in T base_weights[],
	in T base_bias[]) {

	constexpr int FLAT_SIZE = IN_SIZE * GRID_SIZE;

	T   basis_flat[IN * GRID];
	int basis_idx = 0;

	for (int i = 0; i < IN_SIZE; i++) {
		for (int j = 0; j < GRID_SIZE; j++) {
			T diff                        = (input[i] - rbf_grid[j]) * rbf_denom_inv;
			basis_flat[i * GRID_SIZE + j] = exp(-diff * diff);
		}
	}

	for (int out_idx = 0; out_idx < OUT_SIZE; out_idx++) {

		T spline_val = T(0.0);

		for (int j = 0; j < FLAT_SIZE; j++) {
			spline_val += basis_flat[j] * spline_weights[out_idx * FLAT_SIZE + j];
		}

		T base_val = base_bias[out_idx];
		for (int i = 0; i < IN_SIZE; i++) {
			base_val += fastkan_base_activation(input[i]) * base_weights[out_idx * IN_SIZE + i];
		}
		output[out_idx] = spline_val + base_val;
	}
}


