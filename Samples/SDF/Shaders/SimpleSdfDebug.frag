#version 450
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec2 fragCoord;

layout(location = 0) out vec4 fragColor;

#include "SDFConstants.h"

layout(push_constant) uniform PushConstants {
	SDFConstants ctx;
};

vec3 paintCircle (vec2 uv, vec2 center, float rad, float width) {
    
    vec2 diff = center-uv;
    float len = length(diff);
    
    float circle = smoothstep(rad-width, rad, len) - smoothstep(rad, rad+width, len);
    return vec3(circle);
}

void main() {
	// vec2 uv = (fragCoord - 0.5 * ctx.resolution.xy) / ctx.resolution.y;
	vec2 uv = (fragCoord - 0.5 );
	uv.x *= ctx.resolution.x / ctx.resolution.y;
	vec2 mouse = (ctx.mouse.xy - 0.5 * ctx.resolution.xy) / ctx.resolution.y;

	// vec3 color = vec3(uv.x, uv.y, 0.0);
	// vec3 color = vec3(mouse.x, mouse.y, 0.0);
	// vec3 color = vec3(1, 0, 0);
	
	vec3 color = vec3(0, 0, 0);
	vec2 center = vec2(0, 0);
	// vec2 center = vec2(0.5, 0.5);
	float radius = 0.3;
	color += paintCircle(uv, center, radius, 0.01);

	color += paintCircle(uv, mouse, 0.03, 0.01);

	fragColor = vec4(pow(color, vec3(1.0 / 2.2)), 1.0);
}