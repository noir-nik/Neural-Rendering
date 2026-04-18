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

float3 xyz2sph_cartesian(float3 cartesian) {
	const float PI     = 3.1415926535897932384626433832795;
	float       radius = length(cartesian);
	float       theta  = acos(clamp(cartesian.z / radius, -1.0, 1.0));
	float       phi    = atan2(cartesian.y, cartesian.x);
	if (phi < 0.0) phi += 2.0 * PI;
	return float3(radius, theta, phi);
}

void rusinkiewicz_transform(in float3 wi, in float3 wo, out float output[6]) {
	// Compute half vector
	float3 half_vec = wi + wo;
	half_vec        = normalize(half_vec);

	// Convert to spherical coordinates
	float3 sph     = xyz2sph(half_vec);
	float  theta_h = sph.y;
	float  phi_h   = sph.z;

	// Basis vectors
	float3 b_i_normal = float3(0.0, 1.0, 0.0);
	float3 normal     = float3(0.0, 0.0, 1.0);

	// Rotate wi to compute difference vector
	float3 wi_rot = rotate_vector(wi, normal, -phi_h);
	float3 diff   = rotate_vector(wi_rot, b_i_normal, -theta_h);

	// Output: [h_x, h_y, h_z, d_x, d_y, d_z]
	output[0] = half_vec.x;
	output[1] = half_vec.y;
	output[2] = half_vec.z;
	output[3] = diff.x;
	output[4] = diff.y;
	output[5] = diff.z;
}

void rusinkiewicz_transform_isotropic(float3 wi, float3 wo, out float output[6]) {
	float3 half_vec = normalize(wi + wo);

	float3 sph     = xyz2sph(half_vec);
	float  theta_h = sph.y;
	float  phi_h   = sph.z;

	float3 binormal = float3(0.f, 1.f, 0.f);
	float3 normal   = float3(0.f, 0.f, 1.f);

	// Fix for isotropic NBRDF
	half_vec    = rotate_vector((half_vec), normal, -phi_h);
	float3 diff = rotate_vector(wi, normal, -phi_h);
	phi_h       = 0.f;

	diff = normalize(rotate_vector(diff, binormal, -theta_h));

	// Output: [h_x, h_y, h_z, d_x, d_y, d_z]
	output[0] = half_vec.x;
	output[1] = half_vec.y;
	output[2] = half_vec.z;
	output[3] = diff.x;
	output[4] = diff.y;
	output[5] = diff.z;
}

static inline void RusinkiewiczTransform(float3 wi, float3 wo, float3* half, float3* diff) {
	(*half) = normalize(wi + wo);

	float3 sph     = xyz2sph(*half);
	float  theta_h = sph.y;
	float  phi_h   = sph.z;

	float3 binormal = float3(0.f, 1.f, 0.f);
	float3 normal   = float3(0.f, 0.f, 1.f);

	// Fix for isotropic NBRDF
	(*half) = rotate_vector((*half), normal, -phi_h);
	(*diff) = rotate_vector(wi, normal, -phi_h);
	phi_h   = 0.f;

	(*diff) = rotate_vector(*diff, binormal, -theta_h);
	(*diff) = normalize(*diff);
	return;
}

func relu<T : __BuiltinFloatingPointType>(T x)->T {
	return max(x, T(0.));
}

uint pcg_hash(uint seed) {
	uint state = seed * 747796405u + 2891336453u;
	uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
	return (word >> 22u) ^ word;
}

float random_pcg(inout uint seed) {
	seed = pcg_hash(seed);
	return float(seed) / 4294967296.0;
}

uint init_random_seed(uint2 pixel, uint frame, uint sample_index) {

	uint seed = pixel.x * 1973u + pixel.y * 9277u + frame * 26699u + sample_index * 6971u;
	seed      = pcg_hash(seed);
	seed      = pcg_hash(seed);
	return seed;
}

float2 random2D(inout uint seed) {
	float r1 = random_pcg(seed);
	float r2 = random_pcg(seed);
	return float2(r1, r2);
}

// Initialize RNG state from pixel coordinates and frame count
uint rng_init0(uint2 pixel, uint frame) {
	// Combine pixel coordinates and frame for unique seed
	uint seed = pixel.x + pixel.y * 8192u + frame * 67108864u;
	return pcg_hash(seed);
}

uint rng_init(uint2 pixel, uint frame) {
	return pcg_hash(pixel.x + pcg_hash(pixel.y + pcg_hash(frame)));
}

// Generate next random number and update state
uint rng_next(inout uint state) {
	state = pcg_hash(state);
	return state;
}

// Generate random float in [0, 1)
float rng_float(inout uint state) {
	return float(rng_next(state)) * (1.0 / 4294967296.0);
}

// Generate random float in [min, max)
float rng_range(inout uint state, float min_val, float max_val) {
	return min_val + rng_float(state) * (max_val - min_val);
}

float3 TangentToWorld(float3 sample_dir, float3 tangent, float3 bitangent, float3 normal) {
	return sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * normal;
}

float3 WorldToTangent(float3 world_dir, float3 tangent, float3 bitangent, float3 normal) {

	float3 tangent_dir;
	tangent_dir.x = dot(world_dir, tangent);
	tangent_dir.y = dot(world_dir, bitangent);
	tangent_dir.z = dot(world_dir, normal);

	return tangent_dir;
}
// Reflect vector around normal
float3 Reflect(float3 incident, float3 normal) {
	return incident - 2.0 * dot(incident, normal) * normal;
}

float rng_float_range(inout uint rng_state, float min_val, float max_val) {
	return min_val + rng_float(rng_state) * (max_val - min_val);
}

float3 EvaluateBlinnPhong(float3 view_dir, float3 light_dir, float3 normal,
						  float3 diffuse_color, float3 specular_color, float shininess) {
	float3 half_vec = normalize(view_dir + light_dir);
	float  NdotL    = max(dot(normal, light_dir), 0.0);
	float  NdotH    = max(dot(normal, half_vec), 0.0);

	float3 diffuse = diffuse_color / 3.14159265359;

	float  spec_factor   = pow(NdotH, shininess);
	float  normalization = (shininess + 2.0) / (2.0 * 3.14159265359);
	float3 specular      = specular_color * normalization * spec_factor;

	return (diffuse + specular) * NdotL;
}

float3 SampleBlinnPhong0(float2 random_val, float shininess, float3 view_tbn, out float pdf) {

	float cos_theta = pow(random_val.x, 1.0 / (shininess + 1.0));
	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));
	float phi       = 2.0 * 3.14159265359 * random_val.y;

	float3 half_vec = float3(
		sin_theta * cos(phi),
		sin_theta * sin(phi),
		cos_theta);

	float3 light_tbn = 2.0 * dot(half_vec, view_tbn) * half_vec - view_tbn;

	pdf = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cos_theta, shininess);

	float HdotL = max(dot(half_vec, light_tbn), 0.0001);
	pdf         = pdf / (4.0 * HdotL);
	return light_tbn;
	// return half_vec;
}

float3 SampleCosineHemisphere0(float2 random_val, out float pdf) {
	float phi       = 2.0 * 3.14159265359 * random_val.x;
	float cos_theta = sqrt(random_val.y);
	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

	pdf = cos_theta / 3.14159265359;

	return float3(
		cos(phi) * sin_theta,
		sin(phi) * sin_theta,
		cos_theta);
}

float3 SampleCosineHemisphere(float2 random_val, out float pdf) {

	float phi = 2.0 * 3.14159265359 * random_val.x;
	float r   = sqrt(random_val.y);

	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(max(0.0, 1.0 - random_val.y));

	pdf = z / 3.14159265359;

	return float3(x, y, z);
}

float3 SampleBlinnPhong(float2 random_val, float shininess, float3 view_tbn, out float pdf) {

	float phi      = 2.0 * 3.14159265359 * random_val.x;
	float cosTheta = pow(random_val.y, 1.0 / (shininess + 1.0));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

	float3 half_vec;
	half_vec.x = sinTheta * cos(phi);
	half_vec.y = sinTheta * sin(phi);
	half_vec.z = cosTheta;

	float3 light_dir = reflect(-view_tbn, half_vec);

	if (light_dir.z <= 0.0) {
		pdf = 0.0;
		return float3(0.0, 0.0, 1.0);
	}

	float pdf_half = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cosTheta, shininess);

	float VdotH = max(dot(view_tbn, half_vec), 0.0001);
	pdf         = pdf_half / (4.0 * VdotH);

	return light_dir;
}

float3 SampleBlinnPhongReflection(float2 random_val, float shininess, float3 view_tbn, out float pdf) {

	float3 reflect_dir = float3(-view_tbn.x, -view_tbn.y, view_tbn.z);

	float phi      = 2.0 * 3.14159265359 * random_val.x;
	float cosTheta = pow(random_val.y, 1.0 / (shininess + 1.0));
	float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

	float3 up        = abs(reflect_dir.z) < 0.999 ? float3(0, 0, 1) : float3(1, 0, 0);
	float3 tangent   = normalize(cross(up, reflect_dir));
	float3 bitangent = cross(reflect_dir, tangent);

	float3 sample_dir;
	sample_dir.x = sinTheta * cos(phi);
	sample_dir.y = sinTheta * sin(phi);
	sample_dir.z = cosTheta;

	float3 light_dir = sample_dir.x * tangent + sample_dir.y * bitangent + sample_dir.z * reflect_dir;

	light_dir = normalize(light_dir);

	if (light_dir.z <= 0.0) {
		pdf = 0.0;
		return float3(0.0, 0.0, 1.0);
	}

	pdf = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cosTheta, shininess);

	return light_dir;
}

float3 SampleGGX(float2 random_val, float roughness, float3 view_tbn, out float pdf) {
	float alpha  = roughness * roughness;
	float alpha2 = alpha * alpha;

	float phi       = 2.0 * 3.14159265359 * random_val.x;
	float cos_theta = sqrt((1.0 - random_val.y) / (1.0 + (alpha2 - 1.0) * random_val.y));
	float sin_theta = sqrt(max(0.0, 1.0 - cos_theta * cos_theta));

	float3 half_vec = float3(
		sin_theta * cos(phi),
		sin_theta * sin(phi),
		cos_theta);

	float3 light_tbn = 2.0 * dot(half_vec, view_tbn) * half_vec - view_tbn;

	float denom = cos_theta * cos_theta * (alpha2 - 1.0) + 1.0;
	float D     = alpha2 / (3.14159265359 * denom * denom);

	pdf = D * cos_theta;

	float HdotL = max(dot(half_vec, light_tbn), 0.0001);
	pdf         = pdf / (4.0 * HdotL);

	return light_tbn;
}

// GGX NDF (Normal Distribution Function) for isotropic materials
float GGX_D(float NdotH, float alpha) {
	float a2 = alpha * alpha;
	float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
	return a2 / (3.14159265359 * d * d);
}

// Smith masking function for GGX
float GGX_G1(float NdotV, float alpha) {
	float a2     = alpha * alpha;
	float NdotV2 = NdotV * NdotV;
	return 2.0 * NdotV / (NdotV + sqrt(a2 + (1.0 - a2) * NdotV2));
}

// Full GGX VNDF importance sampling for isotropic materials
// Based on "Sampling the GGX Distribution of Visible Normals" by Heitz (2018)
float3 SampleGGXVNDF(float3 view_local, float alpha, float2 u, out float pdf) {
	// Input view_local is in tangent space (z-up)
	// Returns sampled normal (half vector) in tangent space

	// Section 3.2: Transform view direction to hemisphere configuration
	float3 Vh = normalize(float3(alpha * view_local.x, alpha * view_local.y, view_local.z));

	// Section 4.1: Orthonormal basis (with special case if view_local is aligned with z-axis)
	float  lensq = Vh.x * Vh.x + Vh.y * Vh.y;
	float3 T1    = lensq > 0.0 ? float3(-Vh.y, Vh.x, 0.0) * rsqrt(lensq) : float3(1.0, 0.0, 0.0);
	float3 T2    = cross(Vh, T1);

	// Section 4.2: Parameterization of the projected area
	float r   = sqrt(u.x);
	float phi = 2.0 * 3.14159265359 * u.y;
	float t1  = r * cos(phi);
	float t2  = r * sin(phi);
	float s   = 0.5 * (1.0 + Vh.z);
	t2        = (1.0 - s) * sqrt(1.0 - t1 * t1) + s * t2;

	// Section 4.3: Reprojection onto hemisphere
	float3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;

	// Section 3.4: Transforming the normal back to the ellipsoid configuration
	float3 Ne = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0, Nh.z)));

	// Compute PDF
	// PDF = G1(V) * D(H) * max(0, dot(V, H)) / V.z
	float NdotH = Ne.z;
	float VdotH = dot(view_local, Ne);

	if (NdotH <= 0.0 || VdotH <= 0.0) {
		pdf = 0.0;
		return float3(0.0, 0.0, 1.0);
	}

	float D  = GGX_D(NdotH, alpha);
	float G1 = GGX_G1(view_local.z, alpha);

	// VNDF PDF: G1(V) * D(H) * max(0, dot(V, H)) / V.z
	pdf = (G1 * D * VdotH) / max(1e-6, view_local.z);

	return Ne;
}

// Evaluate PDF for cosine-weighted hemisphere sampling
float CosinePDF(float cos_theta) {
	return cos_theta / 3.14159265359;
}

// Evaluate PDF for GGX VNDF sampling
float GGXVNDF_PDF(float3 view_local, float3 half_local, float3 light_local, float alpha) {
	float NdotH = half_local.z;
	float VdotH = dot(view_local, half_local);

	if (NdotH <= 0.0 || VdotH <= 0.0 || view_local.z <= 0.0) {
		return 0.0;
	}

	float D  = GGX_D(NdotH, alpha);
	float G1 = GGX_G1(view_local.z, alpha);

	// VNDF PDF: G1(V) * D(H) * max(0, dot(V, H)) / V.z
	// Then convert from half-vector PDF to light direction PDF
	// pdf_light = pdf_half / (4 * dot(V, H))
	float pdf_half = (G1 * D * VdotH) / max(1e-6, view_local.z);
	return pdf_half / (4.0 * VdotH);
}

float dGgx(float dotNH, float roughness) {
	const float PI = 3.1415926535;

	float alpha  = roughness * roughness;
	float alpha2 = alpha * alpha;
	float denom  = dotNH * dotNH * (alpha2 - 1.0) + 1.0;
	return alpha2 / (PI * denom * denom);
}

float schlickSmithGGX(float dotN, float k) {
	return dotN / (dotN * (1.0 - k) + k);
}
float schlickSmithGGX2(float dotNL, float dotNV, float roughness) {
	float alpha = roughness;
	float k     = (alpha * alpha) / 2.0; // special remap of k for IBL lighting
	float GL    = schlickSmithGGX(dotNL, k);
	float GV    = schlickSmithGGX(dotNV, k);
	return GL * GV;
}

float vdcSequence(uint bits) {
	bits = (bits << 16u) | (bits >> 16u);
	bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
	bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
	bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
	bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
	return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

float2 hammersleySequence(uint i, uint N) {
	return float2(float(i) / float(N), vdcSequence(i));
}

// GGX NDF via importance sampling
float3 importanceSampleGGX(float2 Xi, float3 N, float roughness) {
	const float PI = 3.1415926535;

	float alpha  = roughness * roughness;
	float alpha2 = alpha * alpha;

	float phi      = 2.0 * PI * Xi.x;
	float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (alpha2 - 1.0) * Xi.y));
	float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

	// from spherical coordinates to cartesian coordinates
	float3 H = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

	// from tangent-space vector to world-space sample vector
	float3 up        = abs(N.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
	float3 tangent   = normalize(cross(up, N));
	float3 bitangent = cross(N, tangent);

	float3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}

// ============================================================================
// Multiple Importance Sampling (MIS) Weight Calculations
// ============================================================================

// Multiple Importance Sampling (MIS) weight calculation
// Using power heuristic with beta = 2 (balance heuristic when beta = 1)
float MISWeight(float pdf_a, float pdf_b, float beta) {
	float pdf_a_pow = pow(pdf_a, beta);
	float pdf_b_pow = pow(pdf_b, beta);
	return pdf_a_pow / (pdf_a_pow + pdf_b_pow);
}

// Balance heuristic - optimal for minimizing variance when PDFs are similar
// Most commonly used MIS heuristic
float MISWeightBalance(float pdf_a, float pdf_b) {
	if (pdf_a + pdf_b == 0.0) {
		return 0.0;
	}
	return pdf_a / (pdf_a + pdf_b);
}

// Power heuristic with exponent beta (typically beta = 2)
// Better variance reduction than balance heuristic in most cases
// Veach's thesis recommends beta = 2
float MISWeightPower(float pdf_a, float pdf_b, float beta) {
	if (pdf_a == 0.0) {
		return 0.0;
	}

	float pdf_a_beta = pow(pdf_a, beta);
	float pdf_b_beta = pow(pdf_b, beta);

	if (pdf_a_beta + pdf_b_beta == 0.0) {
		return 0.0;
	}

	return pdf_a_beta / (pdf_a_beta + pdf_b_beta);
}

// Power heuristic with beta = 2 (most common)
float MISWeightPowerSquared(float pdf_a, float pdf_b) {
	if (pdf_a == 0.0) {
		return 0.0;
	}

	float pdf_a_sq = pdf_a * pdf_a;
	float pdf_b_sq = pdf_b * pdf_b;

	if (pdf_a_sq + pdf_b_sq == 0.0) {
		return 0.0;
	}

	return pdf_a_sq / (pdf_a_sq + pdf_b_sq);
}

// Multi-sample MIS weight (for combining more than 2 sampling strategies)
// n_samples[i] = number of samples taken from strategy i
// pdf[i] = PDF of current sample under strategy i
float MISWeightBalanceMulti(float pdf_current, float n_current, float pdfs[], uint num_strategies, float n_samples[]) {
	float sum_weighted_pdfs = 0.0;

	for (uint i = 0; i < num_strategies; i++) {
		sum_weighted_pdfs += n_samples[i] * pdfs[i];
	}

	if (sum_weighted_pdfs == 0.0) {
		return 0.0;
	}

	return (n_current * pdf_current) / sum_weighted_pdfs;
}

// Multi-sample power heuristic
float MISWeightPowerMulti(float pdf_current, float n_current, float pdfs[], uint num_strategies, float n_samples[], float beta) {
	float weighted_pdf_current   = n_current * pdf_current;
	float sum_weighted_pdfs_beta = 0.0;

	for (uint i = 0; i < num_strategies; i++) {
		float weighted_pdf = n_samples[i] * pdfs[i];
		sum_weighted_pdfs_beta += pow(weighted_pdf, beta);
	}

	if (sum_weighted_pdfs_beta == 0.0) {
		return 0.0;
	}

	return pow(weighted_pdf_current, beta) / sum_weighted_pdfs_beta;
}

// #region PDF
//  Calculate PDF of cosine hemisphere sampling for a given direction
float PDFCosineHemisphere(float3 direction_tbn) {
	// direction_tbn.z is cos(theta)
	if (direction_tbn.z <= 0.0) {
		return 0.0;
	}
	return direction_tbn.z / 3.14159265359;
}

// Calculate PDF of Blinn-Phong sampling for a given light direction
float PDFBlinnPhong(float3 view_tbn, float3 light_tbn, float shininess) {
	if (light_tbn.z <= 0.0) {
		return 0.0;
	}

	// Calculate half-vector
	float3 half_vec = normalize(view_tbn + light_tbn);

	if (half_vec.z <= 0.0) {
		return 0.0;
	}

	float VdotH = max(dot(view_tbn, half_vec), 0.0001);

	// PDF of half-vector
	float cosTheta_h = half_vec.z;
	float pdf_half   = (shininess + 1.0) / (2.0 * 3.14159265359) * pow(cosTheta_h, shininess);

	// Convert to light direction PDF
	float pdf_light = pdf_half / (4.0 * VdotH);

	return pdf_light;
}

// #endregion


