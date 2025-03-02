#include "SDFConstants.h"

import SDFScalar;
import SDFVector;
import SDFScalarBuffer;

// clang-format off

[[vk::binding(0, 0)]] StructuredBuffer<float> gSDFWeights;

[[vk::push_constant]] SDFConstants ctx;

func RotateAxisAngle(float3 vector, float3 axis, float angle) -> float3 {
	return lerp(dot(vector, axis) * axis, vector, cos(angle)) + sin(angle) * cross(axis, vector);
}

func LinearOp<T : __BuiltinFloatingPointType, int M, int K>(
	CoopVec<T, K>                  input,
	StructuredBuffer<T>            matrixBiasBuffer,
	uint                           matrixOffset,
	int                            biasOffset,
	constexpr CoopVecMatrixLayout  matrixLayout,
	constexpr CoopVecComponentType componentType,
	uint matrixStride) -> CoopVec<T, M> {
	return coopVecMatMulAdd<T, M>(
		input,
		componentType,
		matrixBiasBuffer,
		matrixOffset,
		componentType,
		matrixBiasBuffer,
		biasOffset,
		componentType,
		matrixLayout,
		false,
		matrixStride);
}

func sin<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input) -> CoopVec<T, N> {
	CoopVec<T, N> output;
	[ForceUnroll]
	for (int i = 0; i < N; ++i) {
		output[i] = sin(input[i]);
	}
	return output;
}

func cos<T : __BuiltinFloatingPointType, let N : int>(CoopVec<T, N> input) -> CoopVec<T, N> {
	CoopVec<T, N> output;
	[ForceUnroll]
	for (int i = 0; i < N; ++i) {
		output[i] = cos(input[i]);
	}
	return output;
}

func SDFCoopVec1(float3 p, StructuredBuffer<float> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4]) -> float {
	CoopVec<float, 3> input;
    CoopVec<float, 16> output0;
    CoopVec<float, 16> output1;
    CoopVec<float, 16> output2;
    CoopVec<float, 1> output3;

	input = {p.x, p.y, p.z};
	// output0 = LinearOp<float, 16, 3>(input, sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16);
	output0 = sin(output0);
	// output1 = LinearOp<float, 16, 16>(output0, sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16);
	output1 = sin(output1) + output0;
	// output2 = LinearOp<float, 16, 16>(output1, sdf_weights, weights_offsets[2], bias_offsets[2], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16);
	output2 = sin(output2) / CoopVec<float, 16>(1.4) + output1;
	// output3 = LinearOp<float, 1, 16>(output2, sdf_weights, weights_offsets[3], bias_offsets[3], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16);

	return output3[0];
}

func SDFCoopVec(float3 p, StructuredBuffer<float> sdf_weights, uint32_t weights_offsets[4], uint32_t bias_offsets[4]) -> float {
	if(length(p) > 1.) {
		return length(p) - .8;
	}

	CoopVec<float, 3>  input0 = {p.x, p.y, p.z};
	CoopVec<float, 16>  output0;
	CoopVec<float, 16>  output1;
	CoopVec<float, 16>  output2;
	CoopVec<float, 1>  output3;

	// for (int i = 0; i < 16; ++i) {
		// output0[i] = sdf_weights[bias_offsets[0] + i];
		// output1[i] = sdf_weights[bias_offsets[1] + i];
		// output2[i] = sdf_weights[bias_offsets[2] + i];
	// }
	// output3[0] = sdf_weights[bias_offsets[3]];

	// output0 = coopVecLoad<16, float>(sdf_weights, bias_offsets[0]);
	// output1 = coopVecLoad<16, float>(sdf_weights, bias_offsets[1]);
	// output2 = coopVecLoad<16, float>(sdf_weights, bias_offsets[2]);
	// output3 = coopVecLoad<1, float>(sdf_weights, bias_offsets[3]);

	output0 = output0.load(sdf_weights, bias_offsets[0]);
	output1 = output1.load(sdf_weights, bias_offsets[1]);
	output2 = output2.load(sdf_weights, bias_offsets[2]);
	output3 = output3.load(sdf_weights, bias_offsets[3]);

	// Forward pass
	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 3; ++j) {
			output0[i] += sdf_weights[weights_offsets[0]/4 + i * 3 + j] * input0[j];
		}
		output0[i] = sin(output0[i]);
	}

	// output0 = LinearOp<float, 16, 3>(input0, sdf_weights, weights_offsets[0], bias_offsets[0], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16, 12);
	// output0 = sin(output0);

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output1[i] += sdf_weights[weights_offsets[1]/4 + i * 16 + j] * output0[j];
		}
		output1[i] = sin(output1[i]) + output0[i];
	}

	// output1 = LinearOp<float, 16, 16>(output0, sdf_weights, weights_offsets[1], bias_offsets[1], CoopVecMatrixLayout::RowMajor, CoopVecComponentType::Float16, 64);
	// output1 = sin(output1) + output0;

	for (int i = 0; i < 16; ++i) {
		for (int j = 0; j < 16; ++j) {
			output2[i] += sdf_weights[weights_offsets[2]/4 + i * 16 + j] * output1[j];
		}
		output2[i] = sin(output2[i]) / 1.4 + output1[i];
	}

	for (int i = 0; i < 16; ++i) {
		output3[0] += sdf_weights[weights_offsets[3]/4 + i] * output2[i];
	}

	return output3[0];
}

func Scene(float3 p) -> float {
	// return SDFCoopVec(p, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	return SDFScalarBuffer(p, gSDFWeights, ctx.weights_offsets, ctx.bias_offsets);
	// return SDFScalar(p);
}

func GetNormal(float3 p) -> float3 {
	return normalize(Scene(p) - float3(Scene(p - float3(0.001, 0, 0)), Scene(p - float3(0, 0.001, 0)), Scene(p - float3(0, 0, 0.001))));
}

[shader("fragment")]
void main(float2 fragCoord, out float4 fragColor: SV_Target0) {
	float2 uv = (fragCoord - 0.5 );
	if (ctx.resolution.x < ctx.resolution.y) {
		uv.y *= ctx.resolution.y / ctx.resolution.x;
	} else {
		uv.x *= ctx.resolution.x / ctx.resolution.y;
	}
	float2 mouse = (ctx.mouse.xy - 0.5 * ctx.resolution.xy) / ctx.resolution.y;
	
	float3 cam = normalize(float3(1.5, uv));
	float3 init = float3(-3., 0, 0);

	float yrot = 0.5;
	float zrot = 0.5;
	zrot = 4. * mouse.x;
	if (true) {
		yrot += -4.*mouse.y;
		zrot = 4.*mouse.x;
	}
	cam = RotateAxisAngle(cam, float3(0, 1, 0), yrot);
	init = RotateAxisAngle(init, float3(0, 1, 0), yrot);
	cam = RotateAxisAngle(cam, float3(0, 0, 1), zrot);
	init = RotateAxisAngle(init, float3(0, 0, 1), zrot);

	float3 p = init;
	bool hit = false;
	for(int i = 0; i < 150 && !hit; ++i) {
		float dist = Scene(p);
		hit = dist * dist < 1e-5;
		p += dist * cam;
		if(distance(p, init) > 5.)
			break;
	}
	// float3 n = GetNormal(p);
	float3 n = D_SDFScalar(p);
	float3 r = reflect(cam, n);
	// float3 nz = p - RotateAxisAngle(p, float3(1, 1, 1), 2.) + RotateAxisAngle(p, float3(1, 1, 1), 4.);
	// float spec = length(sin(r*3.5+sin(nz*120.)*.15)*.4+.6)/sqrt(3.);
	float spec = length(sin(r * 3.5) * .4 + .6) / sqrt(3.);

	spec *= smoothstep(-.3, .2, Scene(p + r * .2));
	float3 col = float3(.1, .1, .12) * spec + pow(spec, 8.);
	// float3 col = n;
	float bgdot = length(sin(cam * 8.) * .4 + .6) / 2.;
	float3 bg = float3(.1, .1, .11) * bgdot + pow(bgdot, 10.);
	fragColor.xyz = hit ? col : bg;
	fragColor = smoothstep(-.02, 1.05, sqrt(fragColor)) * (1. - dot(uv, uv) * .5);
}