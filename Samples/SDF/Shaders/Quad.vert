#version 450

const vec3 vertices[6] = {
    vec3(-1.0, -1.0,  0.0),
    vec3(-1.0,  1.0,  0.0),
    vec3( 1.0, -1.0,  0.0),
    vec3(-1.0,  1.0,  0.0),
    vec3( 1.0,  1.0,  0.0),
    vec3( 1.0, -1.0,  0.0)
};

layout(location = 0) out vec2 fragCoord;

void main() {
    vec3 fragPos = vertices[gl_VertexIndex];
    fragCoord = (fragPos*0.5 + 0.5).xy;
    gl_Position = vec4(fragPos, 1.0);
}