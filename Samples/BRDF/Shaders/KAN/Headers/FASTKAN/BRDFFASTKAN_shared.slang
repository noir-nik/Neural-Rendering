inline void rbf_forward<T : __BuiltinFloatingPointType>(in T x, in T grid[], in T denom_inv, out T basis[], int grid_size) {
	for (int i = 0; i < grid_size; i++) {
		T diff   = (x - grid[i]) * denom_inv;
		basis[i] = exp(-diff * diff);
	}
}

inline T spline_linear_forward<T : __BuiltinFloatingPointType>(in T basis_flat[], int weight_cols /* == basis_size */, in T weights[], int out_idx) {
	T sum = T(0.0);
	for (int i = 0; i < weight_cols; i++) {
		sum += basis_flat[i] * weights[out_idx * weight_cols + i];
	}
	return sum;
}

T expm10<T : __BuiltinFloatingPointType>(T x) {
	return pow(T(1.0) + exp(T(10.0) * x), T(0.1)) - T(1.0);
}

T reluexpm1<T : __BuiltinFloatingPointType>(T x) {
	return max(T(0.0), exp(x) - T(1.0));
}

T silu<T : __BuiltinFloatingPointType>(T x) {
	return x / (T(1.0) + exp(-x));
}

T fastkan_base_activation<T : __BuiltinFloatingPointType>(T x) {
	return silu(x);
}


