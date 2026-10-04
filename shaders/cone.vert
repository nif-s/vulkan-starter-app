#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 vertexColor;

layout(set = 0, binding = 0) uniform ObjectData {
    mat4 mvp;
    vec4 color;
} objectData;

void main() {
    gl_Position = objectData.mvp * vec4(inPosition, 1.0);
    vertexColor = inColor;
}