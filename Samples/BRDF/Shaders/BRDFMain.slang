#include "BRDFConfig.h"
#include "BRDFConstants.h"

// clang-format off

[[vk::push_constant]]
BRDFConstants ctx;

[SpecializationConstant]
const int shader_type = 0;

[[shader("vertex")]]
void vs_main(float3 pos: POSITION, out float4 clipPos: SV_Position) {
}

[shader("fragment")]
void ps_main(float2 fragCoord, out float4 fragColor: SV_Target0) {
 
}

// clang-format on