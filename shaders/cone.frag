#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

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

struct PointLight {
    vec4 position;
    vec4 color; // Цвет RGB и интенсивность
};
struct SpotLight {
    vec4 position;
    vec4 color;
    vec4 direction;
    vec4 angles; // Косинусы внутреннего и внешнего полууглов
};
layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight lights[];
} points;
layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight lights[];
} spots;

vec3 safeNormalize(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 1e-8));
}

vec3 blinnPhong(vec3 N, vec3 V, vec3 L, vec3 radiance, vec3 albedo) {
    float diffuse = max(dot(N, L), 0.0);
    if (diffuse <= 0.0) return vec3(0.0);
    vec3 H = safeNormalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), objectData.parameters.x);
    return radiance * (albedo * diffuse + objectData.specular.rgb * specular);
}

void main() {
    vec3 N = safeNormalize(worldNormal);
    vec3 V = scene.counts.z != 0
        ? safeNormalize(scene.cameraPosition.xyz - worldPosition)
        : -scene.cameraDirection.xyz;
    vec3 albedo = objectData.albedo.rgb;
    if (objectData.parameters.y > 0.5) albedo *= vertexColor;

    vec3 result = albedo * scene.ambient.rgb * scene.ambient.a;
    result += blinnPhong(N, V, -scene.lightDirection.xyz,
        scene.lightColor.rgb * scene.lightColor.a, albedo);
    for (int i = 0; i < scene.counts.x; ++i) {
        vec3 delta = points.lights[i].position.xyz - worldPosition;
        float distanceSquared = max(dot(delta, delta), 1e-4);
        vec3 radiance = points.lights[i].color.rgb * points.lights[i].color.a / distanceSquared;
        result += blinnPhong(N, V, safeNormalize(delta), radiance, albedo);
    }
    for (int i = 0; i < scene.counts.y; ++i) {
        vec3 delta = spots.lights[i].position.xyz - worldPosition;
        vec3 L = safeNormalize(delta);
        // L направлен к источнику, а направление прожектора — от источника.
        float theta = dot(-L, spots.lights[i].direction.xyz);
        float inner = spots.lights[i].angles.x;
        float outer = spots.lights[i].angles.y;
        float cone = inner > outer ? smoothstep(outer, inner, theta) : step(outer, theta);
        float distanceSquared = max(dot(delta, delta), 1e-4);
        vec3 radiance = spots.lights[i].color.rgb * spots.lights[i].color.a * cone / distanceSquared;
        result += blinnPhong(N, V, L, radiance, albedo);
    }
    outColor = vec4(result, 1.0);
}
