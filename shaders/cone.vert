#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;
layout(location = 2) out vec3 vertexColor;

layout(std140, set = 0, binding = 0) uniform SceneData {
    mat4 viewProjection;
    vec4 cameraPosition;
    vec4 cameraDirection;
    vec4 ambient;
    vec4 lightDirection;
    vec4 lightColor;
    ivec4 counts;
} scene;

layout(std140, set = 1, binding = 0) uniform ObjectData {
    mat4 model;
    mat4 normalMatrix;
    vec4 albedo;
    vec4 specular;
    vec4 parameters;
} objectData;

void main() {
    vec4 position = objectData.model * vec4(inPosition, 1.0);
    worldPosition = position.xyz;
    worldNormal = mat3(objectData.normalMatrix) * inNormal;
    vertexColor = inColor;
    gl_Position = scene.viewProjection * position;
}
