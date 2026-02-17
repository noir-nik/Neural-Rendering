#include "BRDFBindings.h"
#include "BRDFConstants.h"

// clang-format off
// struct VSInput {
//     [[vk::location(0)]] float3 position : POSITION0;
// };

struct VSOutput {
    [[vk::location(0)]]
    float4 position : SV_POSITION;
    [[vk::location(1)]]
    float3 view_ray : TEXCOORD0;
};

// Push constants for camera data
// struct EnvMapConstants {
    // float4x4 inv_view_proj;  // Inverse of (projection * view) matrix
// };

[[vk::push_constant]] BRDFConstants ctx;

// [[vk::push_constant]]
// EnvMapConstants constants;

// [[vk::binding(0, 0)]]
// SamplerCube gEnvMap;

[[vk::binding(BINDING_TEXTURE, 0)]] SamplerCube gEnvMap;

// ============================================================================
// Vertex Shader - Fullscreen Triangle
// ============================================================================

static const float3 tri[3] = {
	float3(-1.0, -1.0, 1.0),
	float3( 3.0, -1.0, 1.0),
	float3(-1.0,  3.0, 1.0)
};

[[shader("vertex")]]
// VSOutput vs_main(uint vertexId: SV_VertexID) {
	VSOutput vs_main(uint vertexId: SV_VertexID) {
    VSOutput output;
    
    // Generate fullscreen triangle vertices
    // Vertex 0: (-1, -1)
    // Vertex 1: ( 3, -1)
    // Vertex 2: (-1,  3)
    // This covers the entire screen with one triangle
    float2 uv = float2(
        (vertexId << 1) & 2,
        vertexId & 2
    );
    
    float2 pos = uv * 2.0 - 1.0;
    
    // Position in NDC (covers entire screen, at far plane)
    output.position = float4(pos, 1.0, 1.0);  // z=1.0 for far plane
    
    // Reconstruct view ray for environment map sampling
    // Transform NDC position to world space direction
    float4 world_pos = mul((ctx.view_proj), float4(pos, 1.0, 1.0));
	world_pos.y *= -1.0;
    output.view_ray = world_pos.xyz / world_pos.w;
    
    return output;
}

// ============================================================================
// Fragment Shader
// ============================================================================

[[shader("fragment")]]
float4 ps_main(VSOutput input) : SV_TARGET {
    // Normalize the view ray to get direction
    float3 direction = normalize(input.view_ray);
    
    // Sample the cube map
    float4 color = gEnvMap.Sample(direction);
    
    // Optional: Apply exposure or tone mapping
    color.rgb *= 1.0; // Adjust exposure as needed
    
    return float4(color.rgb, 1.0);
}