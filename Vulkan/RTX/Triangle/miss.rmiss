#version 460
#extension GL_NV_ray_tracing : require

struct RayPayload
{
    vec3 color;
};

layout(location = 0) rayPayloadInNV RayPayload PrimaryRay;

void main()
{
    const vec3 backgroundColor = vec3(0., 0., 0.);
    PrimaryRay.color = backgroundColor;
}
