#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
	T elements[Num ? Num : 1];

	thread T& operator [] (size_t pos) thread
	{
		return elements[pos];
	}
	constexpr const thread T& operator [] (size_t pos) const thread
	{
		return elements[pos];
	}

	device T& operator [] (size_t pos) device
	{
		return elements[pos];
	}
	constexpr const device T& operator [] (size_t pos) const device
	{
		return elements[pos];
	}

	constexpr const constant T& operator [] (size_t pos) const constant
	{
		return elements[pos];
	}

	threadgroup T& operator [] (size_t pos) threadgroup
	{
		return elements[pos];
	}
	constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
	{
		return elements[pos];
	}
};

constant spvUnsafeArray<float2, 6> _TexCoords = spvUnsafeArray<float2, 6>({ float2(0.0), float2(1.0, 0.0), float2(0.0, 1.0), float2(1.0, 0.0), float2(1.0), float2(0.0, 1.0) });
constant spvUnsafeArray<float3, 6> _Vertices = spvUnsafeArray<float3, 6>({ float3(-1.0, -1.0, 0.0), float3(1.0, -1.0, 0.0), float3(-1.0, 1.0, 0.0), float3(1.0, -1.0, 0.0), float3(1.0, 1.0, 0.0), float3(-1.0, 1.0, 0.0) });

struct Vertex
{
	float2 uv [[user(locn1)]];
	float4 position [[position]];
};

vertex Vertex VSMain (uint id [[vertex_id]])
{
	Vertex out = {};
	out.uv = _TexCoords[id];
	out.position = float4(_Vertices[id], 1.0);
	return out;
}

static inline __attribute__((always_inline))
float Line(thread const float2& p, thread const float2& a, thread const float2& b)
{
	float2 pa = p - a;
	float2 ba = b - a;
	float h = fast::max(0.0, fast::min(1.0, dot(pa, ba) / dot(ba, ba)));
	float2 d = pa - (ba * h);
	return dot(d, d);
}

fragment float4 PSMain (Vertex in [[stage_in]])
{
	float k = Line(in.uv, float2(0.3, 0.1), float2(0.8, 0.5));
	return mix(float4(1.0), float4(0.0, 0.0, 0.0, 1.0), float4(smoothstep(0.0, 0.00001, k)));
}