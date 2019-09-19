// cl VolumeParticleSystem.c opengl32.lib user32.lib gdi32.lib

#include <windows.h>
#include <GL/gl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define ScreenWidth 1920.0f
#define ScreenHeight 1080.0f
#define FieldOfView 60.0f
#define NearClip 0.01f
#define FarClip 1000.0f
#define VerticalSync 0

#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_SHADING_LANGUAGE_VERSION       0x8B8C
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_CLAMP_TO_EDGE                  0x812F
#define GL_TEXTURE_CUBE_MAP               0x8513
#define GL_TEXTURE_CUBE_MAP_POSITIVE_X    0x8515
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_X    0x8516
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Y    0x8517
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Y    0x8518
#define GL_TEXTURE_CUBE_MAP_POSITIVE_Z    0x8519
#define GL_TEXTURE_CUBE_MAP_NEGATIVE_Z    0x851A
#define GL_TEXTURE_WRAP_R                 0x8072
#define GL_TEXTURE_MAX_ANISOTROPY         0x84FE
#define GL_RENDERBUFFER                   0x8D41
#define GL_FRAMEBUFFER                    0x8D40
#define GL_DEPTH_ATTACHMENT               0x8D00
#define GL_DEPTH_COMPONENTS               0x8284
#define GL_COLOR_ATTACHMENT0              0x8CE0
#define GL_ARRAY_BUFFER                   0x8892
#define GL_STATIC_DRAW                    0x88E4
#define GL_TEXTURE0                       0x84C0

typedef GLuint (__stdcall *PFNGLCREATEPROGRAMPROC) ();
typedef GLuint (__stdcall *PFNGLCREATESHADERPROC) (GLenum t);
typedef void (__stdcall *PFNGLSHADERSOURCEPROC) (GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void (__stdcall *PFNGLCOMPILESHADERPROC) (GLuint s);
typedef void (__stdcall *PFNGLATTACHSHADERPROC) (GLuint p, GLuint s);
typedef void (__stdcall *PFNGLLINKPROGRAMPROC) (GLuint p);
typedef void (__stdcall *PFNGLUSEPROGRAMPROC) (GLuint p);
typedef int (__stdcall *PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef int (__stdcall *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (__stdcall *PFNGLUNIFORMMATRIX4FVPROC) (GLint l, GLsizei c, GLboolean t, const GLfloat *v);
typedef void (__stdcall *PFNGLUNIFORM3FPROC) (GLint location, float v0, float v1, float v2);
typedef void (__stdcall *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void (__stdcall *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);
typedef void (__stdcall *PFNGLGENFRAMEBUFFERSPROC) (GLsizei n, GLuint *f);
typedef void (__stdcall *PFNGLBINDFRAMEBUFFEREXTPROC) (GLenum t, GLuint f);
typedef void (__stdcall *PFNGLFRAMEBUFFERTEXTUREPROC) (GLenum t, GLenum a, GLuint s, GLint l);
typedef void (__stdcall *PFNGLUNIFORM1IPROC) (GLint l, GLint v);
typedef void (__stdcall *PFNGLGENRENDERBUFFERSPROC) (GLsizei n, GLuint *rend);
typedef void (__stdcall *PFNGLBINDRENDERBUFFERPROC) (GLenum t, GLuint rend);
typedef void (__stdcall *PFNGLRENDERBUFFERSTORAGEPROC) (GLenum t, GLenum i, GLsizei w, GLsizei h);
typedef void (__stdcall *PFNGLFRAMEBUFFERRENDERBUFFERPROC) (GLenum t, GLenum a, GLenum rt, GLuint rb);
typedef void (__stdcall *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void (__stdcall *PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void (__stdcall *PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef void (__stdcall *PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void (__stdcall *PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void (__stdcall *PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void (__stdcall *PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void (__stdcall *PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);
typedef void (__stdcall *PFNGLACTIVETEXTUREPROC) (GLenum texture);
typedef void (__stdcall *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);

PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORMMATRIX4FVPROC glUniformMatrix4fv;
PFNGLUNIFORM3FPROC glUniform3f;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebuffer;
PFNGLFRAMEBUFFERTEXTUREPROC glFramebufferTexture;
PFNGLUNIFORM1IPROC glUniform1i;
PFNGLGENRENDERBUFFERSPROC glGenRenderbuffers;
PFNGLBINDRENDERBUFFERPROC glBindRenderbuffer;
PFNGLRENDERBUFFERSTORAGEPROC glRenderbufferStorage;
PFNGLFRAMEBUFFERRENDERBUFFERPROC glFramebufferRenderbuffer;
PFNGLGENVERTEXARRAYSPROC glGenVertexArrays;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLBINDVERTEXARRAYPROC glBindVertexArray;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray;
PFNGLVERTEXATTRIBPOINTERPROC glVertexAttribPointer;
PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray;
PFNGLACTIVETEXTUREPROC glActiveTexture;
PFNGLUNIFORM1FPROC glUniform1f;
	
float CameraRotationMatrix[4][4], ViewMatrix[4][4], ProjectionViewMatrix[4][4], MVP[4][4], iMouse[2] = {0.0f,0.0f};
unsigned int framebuffer, colormap, depthmap;

void glInit()
{
	glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
	glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
	glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
	glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
	glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
	glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
	glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
	glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
	glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
	wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");	
	glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
	glUniform3f = (PFNGLUNIFORM3FPROC)wglGetProcAddress("glUniform3f");
	glUniformMatrix4fv = (PFNGLUNIFORMMATRIX4FVPROC)wglGetProcAddress("glUniformMatrix4fv");	
	glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers");
	glBindFramebuffer = (PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer");	
	glFramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC)wglGetProcAddress("glFramebufferTexture");	
	glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");	
	glGenRenderbuffers = (PFNGLGENRENDERBUFFERSPROC)wglGetProcAddress("glGenRenderbuffers");	
	glBindRenderbuffer = (PFNGLBINDRENDERBUFFERPROC)wglGetProcAddress("glBindRenderbuffer");	
	glRenderbufferStorage = (PFNGLRENDERBUFFERSTORAGEPROC)wglGetProcAddress("glRenderbufferStorage");	
	glFramebufferRenderbuffer = (PFNGLFRAMEBUFFERRENDERBUFFERPROC)wglGetProcAddress("glFramebufferRenderbuffer");
	glGenVertexArrays = (PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays");
	glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
	glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
	glBindVertexArray = (PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray");
	glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
	glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray");
	glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer");
	glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray");
	glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
	glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");
}

static const char* GroundVertexShader = \
	"#version 460 \n"
	"out vec2 UV;"
	"const vec3 vertices[6] = {vec3(-100,0,-100), vec3(100,0,-100), vec3(-100,0,100), vec3(100,0,-100), vec3(100,0,100), vec3(-100,0,100)};"
	"const vec2 uv[6] = {vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,0), vec2(1,1), vec2(0,1)};"
	"uniform mat4 MVP;"
	
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"UV = uv[id];"
		"gl_Position = MVP * vec4(vertices[id], 1.0);"
	"}";

static const char* GroundFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"	
	"in vec2 UV;"	

	"void main()"
	"{"	
		"fragColor = vec4(1.0);"
	"}";
	
static const char* ParticlesVertexShader = \
	"#version 460 \n"	

	"out vec3 position;"	
	"out vec3 normal;"
	"out float instance;"
	"out vec3 finalcolor;"	
	"uniform float _Timer;"
	"uniform vec3 _EmitterPosition;"	
	"uniform mat4 MVP;"

	"mat3 rotationX( float x) "
	"{"
		"return mat3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));"
	"}"
	
	"mat3 rotationY( float y) "
	"{"
		"return mat3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));"
	"}"

	"mat3 rotationZ( float z) "
	"{"
		"return mat3(cos(z),sin(z),0.0,-sin(z),cos(z),0.0,0.0,0.0,1.0);"
	"}"
			
	"vec3 hash(uint p)"
	"{"
		"p = 1103515245U*((p >> 1U)^(p));"
		"uint h32 = 1103515245U*((p)^(p>>3U));"
		"uint n = h32^(h32 >> 16);"
		"uvec3 rz = uvec3(n, n*16807U, n*48271U);"
		"return vec3((rz >> 1) & uvec3(0x7fffffffU,0x7fffffffU,0x7fffffffU))/float(0x7fffffff);"
	"}"

	"float remap (float x, float a, float b, float c, float d) " 
	"{"
		"return (x-a)/(b-a)*(d-c) + c; "
	"}"
	
	"vec3 Parabola(vec3 start, vec3 end, float height, float t)"
	"{"
		"float p = t * 2.0 - 1.0;"
		"vec3 d = end - start;"
		"vec3 s = start + t * d;"
		"s.y += ( -p * p + 1.0 ) * height;"
		"return s;"
	"}"
			
	"void GenerateCube (inout uint id, inout vec3 normal, inout vec3 position, inout float instance)"
	"{"
		"float PI = 3.14159265;"
		"float q = floor((float(id)-36.0*floor(float(id)/36.0))/6.0); "
		"float s = q-3.0*floor(q/3.0); "
		"float inv = -2.0*step(2.5,q)+1.0;"
		"float f = float(id)-6.0*floor(float(id)/6.0);"
		"float t = f-floor(f/3.0); "
		"float a = (t-6.0*floor(t/6.0))*PI*0.5+PI*0.25;"
		"vec3 p = vec3(cos(a),0.707106781,sin(a))*inv;"
		"float x = (s-2.0*floor(s/2.0))*PI*0.5; "
		"mat4 mat = mat4(1,0,0,0,0,cos(x),sin(x),0,0,-sin(x),cos(x),0,0,0,0,1);"
		"float z = step(2.0,s)*PI*0.5;"
		"mat = mat * mat4(cos(z),-sin(z),0,0,sin(z),cos(z),0,0,0,0,1,0,0,0,0,1);"
		"position = (mat * vec4(p,1.0)).xyz;"
		"normal = (mat*vec4(vec3(0,1,0)*inv,0)).xyz;"
		"instance = floor(float(id)/36.0);"
	"}"
	
	"void main()"
	"{"	
		"int _Amount = 16;"
		"float _Speed = 0.5;"
		"float _Spread = 100.0;"
		"float _ParticleOpacity = 0.0;"
		"float _Height = 100.0;"
		"float _ParticleScale = 0.3;"
		"float _Lifetime = 0.2;"
		"float _DiffuseShading = 0.0;"
		"vec4 _StartColor = vec4(1,1,0,1);"
		"vec4 _MiddleColor = vec4(1,0,0,1);"
		"vec4 _EndColor = vec4(0,0,0,1);"
		"int _EmitterCycles = 1000;"
		"uint id = gl_VertexID;"
		"GenerateCube (id, normal, position, instance);"
		"position = rotationY(_Timer*1.7-instance) * rotationX(_Timer*3.0+instance) * position;"
		"int x = int(instance) % _Amount;"
		"int y = (int(instance) / _Amount) % _Amount;"
		"int z = int(instance) / (_Amount*_Amount);"
		"vec3 uv = (vec3(x,y,z) / float(_Amount)) * 2.0 - 1.0 ;"
		"float factor = float(instance) / (_Amount * _Amount * _Amount);"
		"float range = remap(_Timer* _Speed  ,0.0,_Lifetime + factor,0.0,1.0);"
		"float h = remap(mod(_Timer* _Speed ,_Lifetime + factor) ,0.0,_Lifetime+ factor,0.0,1.0);"
		"finalcolor = mix(mix(_StartColor.rgb, _MiddleColor.rgb, h/0.5), mix(_MiddleColor.rgb, _EndColor.rgb, (h - 0.5)/(1.0 - 0.5)), step(0.5, h));"
		"float spread = remap(_Spread*hash(uint(instance)).x,0.0,_Spread,-_Spread,_Spread);"
		"position += Parabola(uv, uv*spread, _Height, mod(_Timer* _Speed + 2.0*(_Lifetime + factor),_Lifetime + factor));"
		"position *= vec3(_ParticleScale,_ParticleScale,_ParticleScale);"
		"position += _EmitterPosition;"
		"position.y = max(position.y,1.0f);"
		"gl_Position = MVP * vec4(position,1.0);"
	"}";

static const char* ParticlesFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"in vec3 position;"
	"in vec3 normal;"	
	"in float instance;"
	"in vec3 finalcolor;"
	
	"void main()"
	"{"	
		"fragColor = vec4(finalcolor,1.0);"	
	"}";
	
static const char* VignetteVertexShader = \
	"#version 460 \n"	
	"out vec2 UV;"
	"const vec3 vertices[6] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(1,1,0), vec3(-1,1,0)};"
	"const vec2 uv[6] = {vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,0), vec2(1,1), vec2(0,1)};"
	
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"UV = uv[id];"
		"gl_Position = vec4(vertices[id], 1);"	
	"}";

static const char *VignetteFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"in vec2 UV;"
	"uniform sampler2D _MainTex;"

	"void main()"
	"{"	
		"vec3 s = texture(_MainTex,UV).rgb;"
		"vec2 uv = UV;"
		"uv *=  1.0 - uv;"  
		"float vig = uv.x*uv.y * 15.0;"
		"vig = pow(vig, 0.25);"	
		"fragColor = vec4(s,1.0) * vec4(vig);"
	"}";	

static const char* SkyboxVertexShader = \
	"#version 460 \n"

	"uniform mat4 MVP;"
	"uniform vec3 offset;"	
	"const vec4 _LightColor0 = vec4(0.9,0.9,0.9,1.0);"
	"const  float _Exposure = 1.0;"
	"const vec3 _GroundColor = vec3(.369, .349, .341);"
	"const float _SunSize = 0.04;"
	"const float _SunSizeConvergence = 5.0;"
	"const vec3 _SkyTint = vec3(.5, .5, .5);"
	"const float _AtmosphereThickness = 1.0; \n"
	"#define OUTER_RADIUS 1.025 \n"
	"#define kRAYLEIGH (mix(0.0, 0.0025, pow(_AtmosphereThickness,2.5))) \n"
	"#define kMIE 0.0010 \n"
	"#define kSUN_BRIGHTNESS 20.0 \n"
	"#define kMAX_SCATTER 50.0 \n"
	"#define MIE_G (-0.990) \n"
	"#define MIE_G2 0.9801 \n"
	"#define SKY_GROUND_THRESHOLD 0.02 \n"
	"#define SKYBOX_COLOR_IN_TARGET_COLOR_SPACE 0 \n"
	"const vec3 ScatteringWavelength = vec3(.65, .57, .475);"
	"const vec3 ScatteringWavelengthRange = vec3(.15, .15, .15);"    
	"const float kOuterRadius = OUTER_RADIUS; "
	"const float kOuterRadius2 = OUTER_RADIUS*OUTER_RADIUS;"
	"const float kInnerRadius = 1.0;"
	"const float kInnerRadius2 = 1.0;"
	"const float kCameraHeight = 0.0001;"
	"const float kHDSundiskIntensityFactor = 15.0;"
	"const float kSunScale = 400.0 * kSUN_BRIGHTNESS;"
	"const float kKmESun = kMIE * kSUN_BRIGHTNESS;"
	"const float kKm4PI = kMIE * 4.0 * 3.14159265;"
	"const float kScale = 1.0 / (OUTER_RADIUS - 1.0);"
	"const float kScaleDepth = 0.25;"
	"const float kScaleOverScaleDepth = (1.0 / (OUTER_RADIUS - 1.0)) / 0.25;"
	"const float kSamples = 2.0;"
	"const vec3 _WorldSpaceLightPos0 = vec3(30,10,20);"
	"out vec3 vertex;"
	"out vec3 groundColor;"
	"out vec3 skyColor;"
	"out vec3 sunColor;"

	"float scale(float inCos)"
	"{"
		"float x = 1.0 - inCos;"
		"return 0.25 * exp(-0.00287 + x*(0.459 + x*(3.83 + x*(-6.80 + x*5.25))));"
	"}"
	
	"void main()"
	"{"	
		"uint id = gl_VertexID;"	
		"float f = id;"
		"float v = f - 6.0 * floor(f/6.0);"
		"f = (f - v) / 6.;"
		"float a = f - 256.0 * floor(f/256.0);"
		"f = (f-a) / 256.;"
		"float b = f-64.;"
		"a += (v - 2.0 * floor(v/2.0));"
		"b += v==2. || v>=4. ? 1.0 : 0.0;"
		"a = a/256.*6.28318;"
		"b = b/256.*6.28318;"
		"vec3 p = vec3(cos(a)*cos(b), sin(b), sin(a)*cos(b)) * 500.0 ;"				
		"gl_Position = MVP * vec4(p+offset, 1.0);"		
		"vec3 kSkyTintInGammaSpace = _SkyTint;"
		"vec3 kScatteringWavelength = mix(ScatteringWavelength-ScatteringWavelengthRange,ScatteringWavelength+ScatteringWavelengthRange,vec3(1,1,1) - kSkyTintInGammaSpace);"
		"vec3 kInvWavelength = 1.0 / (pow(kScatteringWavelength, vec3(4.0)));"
		"float kKrESun = kRAYLEIGH * kSUN_BRIGHTNESS;"
		"float kKr4PI = kRAYLEIGH * 4.0 * 3.14159265;"
		"vec3 cameraPos = vec3(0,kInnerRadius + kCameraHeight,0);"
		"vec3 eyeRay = normalize(p);"
		"float far = 0.0;"
		"vec3 cIn, cOut;"
		"if(eyeRay.y >= 0.0)"
		"{"
			"far = sqrt(kOuterRadius2 + kInnerRadius2 * eyeRay.y * eyeRay.y - kInnerRadius2) - kInnerRadius * eyeRay.y;"
			"vec3 pos = cameraPos + far * eyeRay;"
			"float height = kInnerRadius + kCameraHeight;"
			"float depth = exp(kScaleOverScaleDepth * (-kCameraHeight));"
			"float startAngle = dot(eyeRay, cameraPos) / height;"
			"float startOffset = depth*scale(startAngle);"
			"float sampleLength = far / kSamples;"
			"float scaledLength = sampleLength * kScale;"
			"vec3 sampleRay = eyeRay * sampleLength;"
			"vec3 samplePoint = cameraPos + sampleRay * 0.5;"
			"vec3 frontColor = vec3(0.0, 0.0, 0.0);"
			"for (int i=0; i<2; i++)"
			"{"
				"float height = length(samplePoint);"
				"float depth = exp(kScaleOverScaleDepth * (kInnerRadius - height));"
				"float lightAngle = dot(normalize(_WorldSpaceLightPos0.xyz), samplePoint) / height;"
				"float cameraAngle = dot(eyeRay, samplePoint) / height;"
				"float scatter = (startOffset + depth*(scale(lightAngle) - scale(cameraAngle)));"
				"vec3 attenuate = exp(-clamp(scatter, 0.0, kMAX_SCATTER) * (kInvWavelength * kKr4PI + kKm4PI));"
				"frontColor += attenuate * (depth * scaledLength);"
				"samplePoint += sampleRay;"
			"}"
			"cIn = frontColor * (kInvWavelength * kKrESun);"
			"cOut = frontColor * kKmESun;"
		"}"
		"else"
		"{"
			"far = (-kCameraHeight) / (min(-0.001, eyeRay.y));"
			"vec3 pos = cameraPos + far * eyeRay;"
			"float cameraScale = scale(dot(-eyeRay, pos));"
			"float lightScale = scale(dot(normalize(_WorldSpaceLightPos0.xyz), pos));"
			"float sampleLength = far / kSamples;"
			"float scaledLength = sampleLength * kScale;"
			"vec3 sampleRay = eyeRay * sampleLength;"
			"vec3 samplePoint = cameraPos + sampleRay * 0.5;"
			"vec3 frontColor = vec3(0.0, 0.0, 0.0);    "           
			"float height = length(samplePoint);"
			"float d = exp(kScaleOverScaleDepth * (kInnerRadius - height));"
			"float scatter = d*(lightScale + cameraScale) - exp((-kCameraHeight) * (1.0/kScaleDepth))*cameraScale;"
			"vec3 attenuate = exp(-clamp(scatter, 0.0, kMAX_SCATTER) * (kInvWavelength * kKr4PI + kKm4PI));"
			"frontColor += attenuate * (d * scaledLength);"
			"samplePoint += sampleRay;"
			"cIn = frontColor * (kInvWavelength * kKrESun + kKmESun);"
			"cOut = clamp(attenuate, 0.0, 1.0);"
		"}"
		"vertex = -(p);"
		"groundColor = _Exposure * (cIn + _GroundColor*_GroundColor * cOut);"
		"skyColor = _Exposure * (cIn * (0.75 + 0.75 * dot(normalize(_WorldSpaceLightPos0.xyz), -eyeRay) * dot(normalize(_WorldSpaceLightPos0.xyz), -eyeRay)));" 
		"float lightColorIntensity = clamp(length(_LightColor0.xyz), 0.25, 1.0);"
		"sunColor = kHDSundiskIntensityFactor * clamp(cOut,0.0,1.0) * _LightColor0.xyz / lightColorIntensity;"			
	"}";

static const char* SkyboxFragmentShader = \
	"#version 460 \n"
	"layout(location = 0) out vec4 fragColor;"		

	"in vec3 vertex;"
	"in vec3 groundColor;"
	"in vec3 skyColor;"
	"in vec3 sunColor;"
	"const float _SunSize = 0.04;"
	"const float _SunSizeConvergence = 5.0; \n"
	"#define MIE_G (-0.990) \n"
	"#define MIE_G2 0.9801 \n"
	"#define SKY_GROUND_THRESHOLD 0.02 \n"
	"const vec3 _WorldSpaceLightPos0 = vec3(30,10,20);"

	"float SunAttenuation(vec3 lightPos, vec3 ray)"
	"{"
		"float EyeCos = pow(clamp(dot(lightPos, ray),0.0,1.0), _SunSizeConvergence);"		
		"float temp = pow(1.0 + MIE_G2 - 2.0 * MIE_G * (-EyeCos), pow(_SunSize,0.65) * 10.);"
		"return (1.5 * ((1.0 - MIE_G2) / (2.0 + MIE_G2)) * (1.0 + EyeCos * EyeCos) / max(temp,1.0e-4));"		
	"}"
	
	"void main()"
	"{"	
		"vec3 color = vec3(0.0, 0.0, 0.0);"
		"vec3 ray = normalize(vertex);"
		"float y = ray.y / SKY_GROUND_THRESHOLD;"
		"color = mix(skyColor, groundColor, clamp(y,0.0,1.0));"
		"if(y < 0.0) color += sunColor * SunAttenuation(normalize(_WorldSpaceLightPos0.xyz), -ray);"
		"fragColor = vec4(sqrt(color),1.0);"
	"}";	

float CameraTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,20.0f,
	0.0f,0.0f,-1.0f,-120.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationYMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationXMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ProjectionMatrix[4][4] = 
{
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,-1.0f,0.0f
};

void DebugShader(int sh)
{
	GLint isCompiled = 0;
	glGetShaderiv(sh,GL_LINK_STATUS,&isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint length = 0;
		glGetShaderiv(sh,GL_INFO_LOG_LENGTH,&length);
		GLsizei q = 0;
		char* log = (char*)malloc(sizeof(char)*length);
		glGetShaderInfoLog(sh,length,&q,log);
		if (length>1)
		{
			FILE *file = fopen ("debug.log","a");
			fprintf (file,"%s\n%s\n",(char*)glGetString(GL_SHADING_LANGUAGE_VERSION),log);
			fclose (file);
			ExitProcess(0);
		}
	}
}

int LoadShaders(const char* VS, const char* FS)
{
	int p = glCreateProgram();
	int sv = glCreateShader(GL_VERTEX_SHADER);
	int sf = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(sv, 1, &VS, 0);
	glShaderSource(sf, 1, &FS, 0);	
	glCompileShader(sv);
	glCompileShader(sf);
	glAttachShader(p,sv);
	glAttachShader(p,sf);
	glLinkProgram(p);
	DebugShader(sv);
	DebugShader(sf);
	return p;
}

void LoadFrameBuffer ()
{
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glGenTextures(1, &colormap);
	glBindTexture(GL_TEXTURE_2D, colormap);
	glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB8, ScreenWidth, ScreenHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colormap, 0);
	glGenTextures(1, &depthmap);  
	glBindTexture(GL_TEXTURE_2D, depthmap); 
	glTexImage2D( GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, ScreenWidth, ScreenHeight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  depthmap, 0);
}

void RenderSkybox (unsigned int shader, float mvp[4][4])
{
	glUseProgram(shader);
	glUniform3f(glGetUniformLocation(shader,"offset"), CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]);
	glUniformMatrix4fv(glGetUniformLocation(shader,"MVP"), 1, GL_FALSE, &mvp[0][0]);
	glDrawArrays(GL_TRIANGLES, 0, 196608);	
}

void RenderParticles(unsigned int shader, float mvp[4][4], float x, float y, float z)
{
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader,"MVP"), 1, GL_FALSE, &mvp[0][0]);
	glUniform1f(glGetUniformLocation(shader, "_Timer"), GetTickCount()*0.001f);
	glUniform3f(glGetUniformLocation(shader, "_EmitterPosition"), x, y, z);
	glDrawArrays(GL_TRIANGLES, 0, 36 * 16 * 16 * 16);
}

void RenderGround (unsigned int shader, float mvp[4][4])
{
	glUseProgram(shader);
	glUniformMatrix4fv(glGetUniformLocation(shader,"MVP"), 1, GL_FALSE, &mvp[0][0]);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

void PostProcessing (unsigned int shader, unsigned int renderbuffer)
{
	glBindTexture(GL_TEXTURE_2D, renderbuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, ScreenWidth, ScreenHeight);
	glUseProgram(shader);
	glUniform1i(glGetUniformLocation(shader, "_MainTex"), 0);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}

float deg2rad(float x) 
{
	return (x * 3.14159265358979323846f / 180.0f);
}

void Mul(float mat1[][4], float mat2[][4], float res[][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res[i][j] = 0;
			for (int k = 0; k < 4; k++) 
			{
				res[i][j] += mat1[i][k]*mat2[k][j];
			}
		}
	}
}

void Inverse( float input[4][4], float k[4][4])
{
	float invOut[16];
	float m[16] = 
	{
		input[0][0],input[0][1],input[0][2],input[0][3],
		input[1][0],input[1][1],input[1][2],input[1][3],
		input[2][0],input[2][1],input[2][2],input[2][3],
		input[3][0],input[3][1],input[3][2],input[3][3]
	};
	float inv[16], det;
	int i;
	inv[0]  =  m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
	inv[4]  = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
	inv[8]  =  m[4] *m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7] *m[9];
	inv[12] = -m[4] *m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6] *m[9];
	inv[1]  = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
	inv[5]  =  m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
	inv[9]  = -m[0] *m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3] *m[9];
	inv[13] =  m[0] *m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2] *m[9];
	inv[2]  =  m[1] *m[6]*m[15]-m[1] *m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2] *m[7]-m[13]*m[3] *m[6];
	inv[6]  = -m[0] *m[6]*m[15]+m[0] *m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2] *m[7]+m[12]*m[3] *m[6];
	inv[10] =  m[0] *m[5]*m[15]-m[0] *m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1] *m[7]-m[12]*m[3] *m[5];
	inv[14] = -m[0] *m[5]*m[14]+m[0] *m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1] *m[6]+m[12]*m[2] *m[5];
	inv[3]  = -m[1] *m[6]*m[11]+m[1] *m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10] -m[9]*m[2] *m[7] +m[9]*m[3] *m[6];
	inv[7]  =  m[0] *m[6]*m[11]-m[0] *m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10] +m[8]*m[2] *m[7] -m[8]*m[3] *m[6];
	inv[11] = -m[0] *m[5]*m[11]+m[0] *m[7]*m[9] +m[4]*m[1]*m[11]-m[4]*m[3] *m[9] -m[8]*m[1] *m[7] +m[8]*m[3] *m[5];
	inv[15] =  m[0] *m[5]*m[10]-m[0] *m[6]*m[9] -m[4]*m[1]*m[10]+m[4]*m[2] *m[9] +m[8]*m[1] *m[6] -m[8]*m[2] *m[5];
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	det = 1.0 / det;
	for (i = 0; i < 16; i++) invOut[i] = inv[i] * det;	
	k[0][0] = invOut[0];  k[0][1] = invOut[1];  k[0][2] = invOut[2];  k[0][3] = invOut[3];
	k[1][0] = invOut[4];  k[1][1] = invOut[5];  k[1][2] = invOut[6];  k[1][3] = invOut[7];
	k[2][0] = invOut[8];  k[2][1] = invOut[9];  k[2][2] = invOut[10]; k[2][3] = invOut[11];
	k[3][0] = invOut[12]; k[3][1] = invOut[13]; k[3][2] = invOut[14]; k[3][3] = invOut[15];  
}

void MouseLook()
{	
	POINT point;
	int mx = (int)ScreenWidth  >> 1;
	int my = (int)ScreenHeight >> 1;
	GetCursorPos(&point);
	if( (point.x == mx) && (point.y == my) ) return;
	SetCursorPos(mx, my);	
	float deltaZ = (float)((mx - point.x)) ;
	float deltaX = (float)((my - point.y)) ;
	if (deltaX>0.0f) iMouse[0]+=1.5f; 
	if (deltaX<0.0f) iMouse[0]-=1.5f; 
	if (deltaZ>0.0f) iMouse[1]+=1.5f; 
	if (deltaZ<0.0f) iMouse[1]-=1.5f; 
	CameraRotationXMatrix[1][1] = cos(deg2rad(iMouse[0]));
	CameraRotationXMatrix[1][2] = (-1.0f)*sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][1] = sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][2] = cos(deg2rad(iMouse[0]));
	CameraRotationYMatrix[0][0] = cos(deg2rad(iMouse[1]));
	CameraRotationYMatrix[0][2] = sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][0] = (-1.0f)*sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][2] = cos(deg2rad(iMouse[1]));
}

void KeyboardMovement()
{
	float dx = 0.0f;
	float dz = 0.0f;
	if (GetAsyncKeyState(0x57) ) dz =  32.0f;
	if (GetAsyncKeyState(0x53) ) dz = -32.0f ;
	if (GetAsyncKeyState(0x44) ) dx =  32.0f;
	if (GetAsyncKeyState(0x41) ) dx = -32.0f ;
	CameraTranslationMatrix[0][3] += (-dz * ViewMatrix[2][0] + dx * ViewMatrix[0][0]) * 0.001f;
	CameraTranslationMatrix[1][3] += (-dz * ViewMatrix[2][1] + dx * ViewMatrix[1][0]) * 0.001f;
	CameraTranslationMatrix[2][3] += (-dz * ViewMatrix[2][2] + dx * ViewMatrix[2][0]) * 0.001f;	
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE))
	{
		PostQuitMessage(0); return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_POPUP, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	ShowCursor(0);
	glInit();
	wglSwapIntervalEXT(VerticalSync);
	int SQ = LoadShaders(GroundVertexShader, GroundFragmentShader);
	int SK = LoadShaders(SkyboxVertexShader, SkyboxFragmentShader);	
	int PS = LoadShaders(ParticlesVertexShader, ParticlesFragmentShader);
	int SV = LoadShaders(VignetteVertexShader, VignetteFragmentShader);	
	ProjectionMatrix[0][0] = ((1.0f/tan(deg2rad(FieldOfView/2.0f)))/(ScreenWidth/ScreenHeight));
	ProjectionMatrix[1][1] = (1.0f/tan(deg2rad(FieldOfView/2.0f)));
	ProjectionMatrix[2][2] = (-1.0f)* (FarClip+NearClip)/(FarClip-NearClip);
	ProjectionMatrix[2][3] = (-1.0f)*(2.0f*FarClip*NearClip)/(FarClip-NearClip);
	LoadFrameBuffer ();	
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		MouseLook();
		Mul(CameraRotationYMatrix,CameraRotationXMatrix,CameraRotationMatrix);
		Mul(CameraTranslationMatrix,CameraRotationMatrix,ViewMatrix);
		Inverse(ViewMatrix,ViewMatrix);
		Mul(ProjectionMatrix,ViewMatrix,MVP);
		float MVPT[4][4] = 
		{
			MVP[0][0], MVP[1][0], MVP[2][0], MVP[3][0],
			MVP[0][1], MVP[1][1], MVP[2][1], MVP[3][1],
			MVP[0][2], MVP[1][2], MVP[2][2], MVP[3][2],
			MVP[0][3], MVP[1][3], MVP[2][3], MVP[3][3]
		};
		KeyboardMovement();
		if (GetAsyncKeyState(0x58) & 0x8000) 
			glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		else 
			glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glViewport(0, 0, ScreenWidth, ScreenHeight);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);	
		RenderSkybox (SK, MVPT);
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );
		RenderGround(SQ, MVPT);
		RenderParticles(PS, MVPT, -50.0f, 0.0f,-50.0f);
		RenderParticles(PS, MVPT,  50.0f, 0.0f,-50.0f);
		RenderParticles(PS, MVPT,  50.0f, 0.0f, 50.0f);
		RenderParticles(PS, MVPT, -50.0f, 0.0f, 50.0f);
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);
		PostProcessing (SV, colormap);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	}
	return 0;
}
