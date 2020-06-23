#version 460
#extension GL_NV_ray_tracing : require

struct RayPayload
{
    vec3 color;
};

layout(location = 0) rayPayloadInNV RayPayload PrimaryRay;
hitAttributeNV vec2 HitAttribs;

void main()
{
    const vec3 barycentrics = vec3(1.0f - HitAttribs.x - HitAttribs.y, HitAttribs.x, HitAttribs.y);
    PrimaryRay.color = vec3(barycentrics);
}
