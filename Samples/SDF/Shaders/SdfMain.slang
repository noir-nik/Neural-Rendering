// #include "SDFConfig.h"
#include "SDFConstants.h"

import SDFVec4;
import SDFCoopVec;
import SDFWeightsInBuffer;
import SDFWeightsInHeader;

import SDFWeights_3_16_16_16_1_625;
import SDFWeights_3_24_24_24_1_1321;
import SDFWeights_3_32_32_32_1_2273;
import SDFWeights_3_32_32_32_32_1_3329;
import SDFWeights_3_48_48_48_1_4945;
import SDFWeights_3_64_64_64_1_8641;
import SDFWeights_3_128_128_128_1_33665;
import SDFWeights_3_128_128_128_128_1_50177;
import SDFWeights_3_128_128_128_128_128_1_66689;

typealias FTy = float;
typealias Ty  = half;

// clang-format off
[[vk::binding(0, 0)]] StructuredBuffer<FTy> gWeights;
[[vk::binding(0, 0)]] StructuredBuffer<Ty> gWeightsF16;
// [[vk::binding(0, 0)]] StructuredBuffer<COOPVEC_TYPE> gWeights;

[[vk::push_constant]] SDFConstants ctx;

[SpecializationConstant] const int shader_type = 0;
[SpecializationConstant] const int func_id = 0;

// clang-format on

func RotateAxisAngle(float3 vector, float3 axis, float angle)->float3 {
	return lerp(dot(vector, axis) * axis, vector, cos(angle)) + sin(angle) * cross(axis, vector);
}

float Scene(float3 pos) {

	// vector<COOPVEC_TYPE, 3> pos = { p.x, p.y, p.z };
	// vector<COOPVEC_TYPE, 3> pos = { p.x, p.y, p.z };

	Ty input[3] = { pos.x, pos.y, pos.z };
	Ty output[1];

	FTy inputf[3] = { pos.x, pos.y, pos.z };
	FTy outputf[1];

	uint wo4[4] = { ctx.weights_offsets[0], ctx.weights_offsets[1], ctx.weights_offsets[2], ctx.weights_offsets[3] };
	uint bo4[4] = { ctx.bias_offsets[0], ctx.bias_offsets[1], ctx.bias_offsets[2], ctx.bias_offsets[3] };

	uint wo5[5] = { ctx.weights_offsets[0], ctx.weights_offsets[1], ctx.weights_offsets[2], ctx.weights_offsets[3], ctx.weights_offsets[4] };
	uint bo5[5] = { ctx.bias_offsets[0], ctx.bias_offsets[1], ctx.bias_offsets[2], ctx.bias_offsets[3], ctx.bias_offsets[4] };

	uint wo6[6] = { ctx.weights_offsets[0], ctx.weights_offsets[1], ctx.weights_offsets[2], ctx.weights_offsets[3], ctx.weights_offsets[4], ctx.weights_offsets[5] };
	uint bo6[6] = { ctx.bias_offsets[0], ctx.bias_offsets[1], ctx.bias_offsets[2], ctx.bias_offsets[3], ctx.bias_offsets[4], ctx.bias_offsets[5] };

	// typealias HTy = half;
	typealias HTy = float;

	HTy inputh[3] = { pos.x, pos.y, pos.z };
	HTy outputh[1];

	// if (shader_type != SdfFunctionType::eWeightsInHeader) {
	// 	return 0.5;
	// }

	constexpr int fi = func_id;

	if (shader_type == SdfFunctionType::eCoopVec) {
		// SDFWeightsInHeader<Ty>(input, output);
		// SDFCoopVec<Ty>(input, output, gWeightsF16, wo4, bo4);
		// SDFCoopVec_3_16_16_16_1_625<Ty>(input, output, gWeightsF16, wo4, bo4);

		// SDFWeightsInHeader_3_16_16_16_1_625<Ty>(input, output);
		// SDFWeightsInHeader_3_24_24_24_1_1321<Ty>(input, output);
		// SDFWeightsInHeader_3_32_32_32_1_2273<Ty>(input, output);
		// SDFWeightsInHeader_3_32_32_32_32_1_3329<Ty>(input, output);
		// SDFWeightsInHeader_3_48_48_48_1_4945<Ty>(input, output);

		if (fi == 0) SDFCoopVec_3_16_16_16_1_625<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 1) SDFCoopVec_3_24_24_24_1_1321<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 2) SDFCoopVec_3_32_32_32_1_2273<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 3) SDFCoopVec_3_32_32_32_32_1_3329<Ty>(input, output, gWeightsF16, wo5, bo5);
		else if (fi == 4) SDFCoopVec_3_48_48_48_1_4945<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 5) SDFCoopVec_3_64_64_64_1_8641<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 6) SDFCoopVec_3_128_128_128_1_33665<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 7) SDFCoopVec_3_128_128_128_128_1_50177<Ty>(input, output, gWeightsF16, wo5, bo5);
		else if (fi == 8) SDFCoopVec_3_128_128_128_128_128_1_66689<Ty>(input, output, gWeightsF16, wo6, bo6);
		else return 0.5;

	} else if (shader_type == SdfFunctionType::eWeightsInBuffer) {
		// SDFWeightsInHeader<FTy>(inputf, outputf);
		// SDFWeightsInHeader_3_16_16_16_1_625<FTy>(inputf, outputf);
		// SDFWeightsInBuffer<FTy>(inputf, outputf, gWeights, wo4, bo4);

		if (fi == 0) SDFWeightsInBuffer_3_16_16_16_1_625<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 1) SDFWeightsInBuffer_3_24_24_24_1_1321<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 2) SDFWeightsInBuffer_3_32_32_32_1_2273<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 3) SDFWeightsInBuffer_3_32_32_32_32_1_3329<FTy>(inputf, outputf, gWeights, wo5, bo5);
		else if (fi == 4) SDFWeightsInBuffer_3_48_48_48_1_4945<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 5) SDFWeightsInBuffer_3_64_64_64_1_8641<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 6) SDFWeightsInBuffer_3_128_128_128_1_33665<FTy>(inputf, outputf, gWeights, wo4, bo4);
		else if (fi == 7) SDFWeightsInBuffer_3_128_128_128_128_1_50177<FTy>(inputf, outputf, gWeights, wo5, bo5);
		else if (fi == 8) SDFWeightsInBuffer_3_128_128_128_128_128_1_66689<FTy>(inputf, outputf, gWeights, wo6, bo6);
		else return 0.5;
	} else if (shader_type == SdfFunctionType::eWeightsInBufferF16) {
		// SDFWeightsInHeader<Ty>(input, output);
		// SDFWeightsInBuffer<Ty>(input, output, gWeightsF16, wo4, bo4);
		// SDFWeightsInBuffer_3_16_16_16_1_625<FTy>(inputf, outputf, gWeights, ctx.weights_offsets, ctx.bias_offsets);
		// SDFWeightsInBuffer_3_64_64_64_1_8641<Ty>(input, output, gWeightsF16, wo4, bo4);

		if (fi == 0) SDFWeightsInBuffer_3_16_16_16_1_625<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 1) SDFWeightsInBuffer_3_24_24_24_1_1321<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 2) SDFWeightsInBuffer_3_32_32_32_1_2273<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 3) SDFWeightsInBuffer_3_32_32_32_32_1_3329<Ty>(input, output, gWeightsF16, wo5, bo5);
		else if (fi == 4) SDFWeightsInBuffer_3_48_48_48_1_4945<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 5) SDFWeightsInBuffer_3_64_64_64_1_8641<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 6) SDFWeightsInBuffer_3_128_128_128_1_33665<Ty>(input, output, gWeightsF16, wo4, bo4);
		else if (fi == 7) SDFWeightsInBuffer_3_128_128_128_128_1_50177<Ty>(input, output, gWeightsF16, wo5, bo5);
		else if (fi == 8) SDFWeightsInBuffer_3_128_128_128_128_128_1_66689<Ty>(input, output, gWeightsF16, wo6, bo6);
		else return 0.5;

	} else if (shader_type == SdfFunctionType::eWeightsInHeader) {
		SDFWeightsInHeader<HTy>(inputh, outputh);

		// SDFWeightsInHeader_3_16_16_16_1_625<HTy>(inputh, outputh);
		// SDFWeightsInHeader_3_24_24_24_1_1321<HTy>(inputh, outputh); // 13.2h // 13f
		// SDFWeightsInHeader_3_32_32_32_1_2273<HTy>(inputh, outputh); // 180h 180f
		// SDFWeightsInHeader_3_32_32_32_32_1_3329<HTy>(inputh, outputh); // 190h // undef f
		// SDFWeightsInHeader_3_48_48_48_1_4945<HTy>(inputh, outputh);
		// SDFWeightsInHeader_3_64_64_64_1_8641<HTy>(inputh, outputh);

		// if (fi == 0) SDFWeightsInHeader_3_16_16_16_1_625<HTy>(inputh, outputh);
		// else if (fi == 1) SDFWeightsInHeader_3_24_24_24_1_1321<HTy>(inputh, outputh);
		// else if (fi == 2) SDFWeightsInHeader_3_32_32_32_1_2273<HTy>(inputh, outputh);
		// else if (fi == 4) SDFWeightsInHeader_3_32_32_32_32_1_3329<HTy>(inputh, outputh);
		// else if (fi == 5) SDFWeightsInHeader_3_48_48_48_1_4945<HTy>(inputh, outputh);
		// else if (fi == 5) SDFWeightsInHeader_3_256_256_256_256_256_1_264449<HTy>(input, output);
		// else if (fi == 5) BRDFWeightsInHeader<Ty>(input, output);
		// else if (fi == 6) BRDFWeightsInHeader<float>(input, output);
		// else SDFWeightsInHeader<HTy>(inputh, outputh);
		// else return 0.5;
	} else if (shader_type == SdfFunctionType::eVec4) {
		// SDFWeightsInHeader<HTy>(inputf, outputf);
		inputf[0] = SDFVec4<FTy>(pos);
	} else {
		// SDFWeightsInHeader<Ty>(input, output);
		SDFCoopVec<Ty>(input, output, gWeightsF16, wo4, bo4);
	}

	if (shader_type == SdfFunctionType::eWeightsInHeader) {
		return outputh[0];
	} else if (shader_type == SdfFunctionType::eCoopVec || shader_type == SdfFunctionType::eWeightsInBufferF16) {
		return output[0];
	} else {
		return outputf[0];
	}
}

func GetNormal(float3 p)->float3 {
	const float kDiffEpsilon = float(0.1);
	return normalize(Scene(p) - float3(Scene(p - float3(kDiffEpsilon, 0, 0)), Scene(p - float3(0, kDiffEpsilon, 0)), Scene(p - float3(0, 0, kDiffEpsilon))));
}

[shader("fragment")]
void main(float2 fragCoord, out float4 fragColor: SV_Target0) {
	const float PI         = 3.141592653589793238462643383279;
	const float DEG_TO_RAD = PI / 180.0f;
	const float kEpsilon   = 2e-3;
	// const float kEpsilon = 5e-3;
	// const float kEpsilon = 1e-2;
	const float kMaxDist = 1000.0f;
	const int kSteps = 100;
	// const int kSteps = 50;

	// fragColor.w = 1.0f;

	float2 uv = (fragCoord - 0.5); // center of the pixel

	float aspect_ratio = ctx.resolution_x / ctx.resolution_y;

	float2 fov = tan(ctx.fov * DEG_TO_RAD);

	if (aspect_ratio > 1) {
		uv.y /= aspect_ratio;
	} else {
		uv.x *= aspect_ratio;
	}

	let factor = uv * fov;

	float3 ray_dir =
		ctx.camera_forward
		+ factor.x * ctx.camera_right
		+ factor.y * ctx.camera_up;

	ray_dir = normalize(ray_dir);

	float3 init = ctx.camera_pos;

	float3 p   = init;
	bool   hit = false;
	// [ForceUnroll(100)]
	for (int i = 0; i < kSteps && !hit; ++i) {
		float dist = Scene(p);

		// hit = dist * dist < kEpsilon;
		hit = dist < kEpsilon;
		p += dist * ray_dir;
		if (distance(p, init) > kMaxDist)
			break;
	}

	float3 n  = GetNormal(p);
	float3 bg = float3(.1, .1, .11);

	{ // float3 n = D_SDFScalar<COOPVEC_TYPE>({p.x, p.y, p.z});
		float3 r    = reflect(ray_dir, n);
		float3 nz   = p - RotateAxisAngle(p, float3(1, 1, 1), 2.) + RotateAxisAngle(p, float3(1, 1, 1), 4.);
		float  spec = length(sin(r * 3.5 + sin(nz * 120.) * .15) * .4 + .6) / sqrt(3.);
		// float spec = length(sin(r * 3.5) * .4 + .6) / sqrt(3.);

		spec *= smoothstep(-.3, .2, Scene(p + r * .2));
		float3 col   = float3(.1, .1, .12) * spec + pow(spec, 8.);
		float  bgdot = length(sin(ray_dir * 8.) * .4 + .6) / 2.;
		float3 bg    = float3(.1, .1, .11) * bgdot + pow(bgdot, 10.);
		bg           = float3(1.0f, 1.0f, 1.0f);

		col = smoothstep(-.02, 1.05, sqrt(col)) * (1. - dot(uv, uv) * .5);

		fragColor.xyz = hit ? col : bg;
		fragColor.w   = 1.0f;
		// if (hit) {
		// 	fragColor.xyz = col;
		// 	fragColor.w = 1.0f;
		// }
		// fragColor     = smoothstep(-.02, 1.05, sqrt(fragColor)) * (1. - dot(uv, uv) * .5);

		return;
	}
	float3 light_pos = float3(-2.0, 2.0, -2.0);

	float3 lightDir      = normalize(light_pos - p);
	float3 viewDir       = normalize(init - p);
	float3 halfwayDir    = normalize(lightDir + viewDir);
	float3 normal        = normalize(n);
	float3 lightColor    = float3(0.8, 0.8, 0.8);
	float3 specularColor = float3(0.8, 0.8, 0.8);
	float  spec; //= pow(max(dot(normal, halfwayDir), 0.0), shininess);
	// float3   specular   = lightColor * spec;
	float3 ambient = float3(.2, .2, .2);

	// float shininess = 1.0;
	float3 col       = {};
	float  shininess = 1;
	shininess        = 64;
	shininess        = 24;
	if (1)
	// if (0)
	{
		float3 halfwayDir = normalize(lightDir + viewDir);
		spec              = pow(max(dot(normal, halfwayDir), 0.0), shininess * 2);
	} else {
		float3 reflectDir = reflect(-lightDir, normal);
		spec              = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
	}

	float3 objectColor = float3(0.8, 0.8, 0.8);
	{
		float3 specular = lightColor * spec;
		float3 diffuse  = lightColor * max(dot(lightDir, normal), 0.0);
		diffuse *= 0.2;
		float3 col = (ambient + diffuse + specular) * objectColor;
	}

	{
		// Reflection vector
		float3 reflectDir = reflect(-lightDir, normal);

		// Ambient term
		float3 ambient = ambient * lightColor;

		// Diffuse term (Lambert)
		float  diff    = max(dot(normal, lightDir), 0.0);
		float3 diffuse = diff * objectColor * lightColor;

		// Specular term (Phong)
		float  spec     = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
		float3 specular = spec * specularColor * lightColor;
		specular *= 0.2;

		float cosViewNormal = dot(viewDir, normal);

		// float fresnel = clamp(1.0 + pow(1.0 - cosViewNormal, 20.0), 0.0, 1.0);
		float fresnel = -cosViewNormal * cosViewNormal + 1.0;
		fresnel       = clamp(pow(1 - fresnel, 0.2), 0.0, 1.0);
		fresnel       = 1.;
		// Combine
		float3 phong = (ambient + diffuse + specular) * fresnel;

		col = phong;
	}

	fragColor.xyz = hit ? col : bg;
	fragColor.w   = 1.0;
}


