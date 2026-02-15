#include "BRDFBindings.h"
#include "BRDFConstants.h"

// clang-format off
// struct VSInput {
//     [[vk::location(0)]] float3 position : POSITION0;
// };

struct VSOutput {
    [[vk::location(0)]] float4 clip_pos : SV_POSITION;
    [[vk::location(1)]] float3 local_pos : TEXCOORD0;
};

// struct SkyboxConstants {
//     float4x4 view_proj;
//     float3 camera_pos;
//     float padding;
// };

// [[vk::push_constant]] SkyboxConstants constants;
[[vk::push_constant]] BRDFConstants ctx;

[[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;
// clang-format on
 
// Cube vertices (generated in code or here)
static const float3 cubeVertices[36] = {
    // Back face
    float3(-1.0, -1.0, -1.0),
    float3( 1.0,  1.0, -1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0,  1.0, -1.0),
    float3(-1.0, -1.0, -1.0),
    float3(-1.0,  1.0, -1.0),
    
    // Front face
    float3(-1.0, -1.0,  1.0),
    float3( 1.0, -1.0,  1.0),
    float3( 1.0,  1.0,  1.0),
    float3( 1.0,  1.0,  1.0),
    float3(-1.0,  1.0,  1.0),
    float3(-1.0, -1.0,  1.0),
    
    // Left face
    float3(-1.0,  1.0,  1.0),
    float3(-1.0,  1.0, -1.0),
    float3(-1.0, -1.0, -1.0),
    float3(-1.0, -1.0, -1.0),
    float3(-1.0, -1.0,  1.0),
    float3(-1.0,  1.0,  1.0),
    
    // Right face
    float3( 1.0,  1.0,  1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0,  1.0, -1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0,  1.0,  1.0),
    float3( 1.0, -1.0,  1.0),
    
    // Bottom face
    float3(-1.0, -1.0, -1.0),
    float3( 1.0, -1.0, -1.0),
    float3( 1.0, -1.0,  1.0),
    float3( 1.0, -1.0,  1.0),
    float3(-1.0, -1.0,  1.0),
    float3(-1.0, -1.0, -1.0),
    
    // Top face
    float3(-1.0,  1.0, -1.0),
    float3( 1.0,  1.0,  1.0),
    float3( 1.0,  1.0, -1.0),
    float3( 1.0,  1.0,  1.0),
    float3(-1.0,  1.0, -1.0),
    float3(-1.0,  1.0,  1.0)
};

[[shader("vertex")]]
VSOutput vs_main(uint vertexId: SV_VertexID) {
    VSOutput output;

	float3 position = cubeVertices[vertexId];
	// Remove translation from view matrix (keep rotation only)
	float4x4 view_no_translation = ctx.view_proj;
    view_no_translation[3][0] = 0.0;
    view_no_translation[3][1] = 0.0;
    view_no_translation[3][2] = 0.0;
    
    // Position at far plane
    float4 pos = mul(view_no_translation, float4(position, 1.0));
    output.clip_pos = pos.xyww; // Set z = w for far plane
    
    output.local_pos = position;
    
    return output;
}

[[shader("fragment")]]
float4 ps_main(VSOutput input) : SV_TARGET {
    float3 direction = normalize(input.local_pos);
    float4 color = gEnvMap.Sample(direction);
    return float4(color.rgb, 1.0);
}