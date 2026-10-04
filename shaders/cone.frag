#version 450

layout(location = 0) in vec3 vertexColor;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 mvp;
    vec4 color;
} objectData;

void main() {
    outColor = vec4(vertexColor, 1.0) * objectData.color;
}