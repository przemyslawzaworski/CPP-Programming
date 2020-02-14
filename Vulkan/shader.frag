// https://www.shadertoy.com/view/4d2BW1
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 fragColor;

layout( push_constant ) uniform TimerBlock
{
	vec4 iTime;
} PushConstant;

float m(vec3 p)
{
	float i = 0., s = 1., k = 0.;
	for(p.y += PushConstant.iTime.x * 0.3 ; i++<7.; s *= k ) p *= k = 1.5 / dot(p = mod(p - 1., 2.) - 1., p);
	return length(p)/s - .01;   
}

void main()
{				
	vec3 d = vec3((2.0*gl_FragCoord.xy-vec2(1920,1080))/1080.,1)/6.0; 
	vec3 o = vec3(1.0);
	vec4 c = vec4(0.0);
	while(c.w++<100.) o+= m(o)*d;       
	c.rgb += m(o - d) / pow(o.z,1.5) * 2.5;
	fragColor = vec4(c.rgb, 1.0);
}