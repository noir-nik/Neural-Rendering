float3 rotate_vector(float3 v, float3 axis, float angle) {
	float c = cos(angle);
	float s = sin(angle);
	return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

float3 xyz2sph(float3 v) {
	float x = v.x, y = v.y, z = v.z;
	float r_xy2 = x * x + y * y;
	float r     = sqrt(r_xy2 + z * z);
	float theta = atan2(sqrt(r_xy2), z);
	float phi   = atan2(y, x);
	return float3(r, theta, phi);
}

void rusinkiewicz_transform(in float3 wi, in float3 wo, out float output[6]) {
	// Compute half vector
	float3 half_vec = wi + wo;
	half_vec = normalize(half_vec);

	// Convert to spherical coordinates
	float3 sph = xyz2sph(half_vec);
	float theta_h = sph.y;
	float phi_h = sph.z;

	// Basis vectors
	float3 b_i_normal = float3(0.0, 1.0, 0.0);
	float3 normal = float3(0.0, 0.0, 1.0);

	// Rotate wi to compute difference vector
	float3 wi_rot = rotate_vector(wi, normal, -phi_h);
	float3 diff = rotate_vector(wi_rot, b_i_normal, -theta_h);

	// Output: [h_x, h_y, h_z, d_x, d_y, d_z]
	output[0] = half_vec.x;
	output[1] = half_vec.y;
	output[2] = half_vec.z;
	output[3] = diff.x;
	output[4] = diff.y;
	output[5] = diff.z;
}
