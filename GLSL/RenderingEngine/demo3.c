// Author: Przemyslaw Zaworski
// cl demo3.c opengl32.lib user32.lib gdi32.lib winmm.lib
// Modo: Geometry -> Polygon -> Triple, then Export Selected Layers -> *.obj (Just Layers)
// Gimp or IrfanView: flip image vertically, then save as binary PPM

#include <windows.h>
#include <GL/gl.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mmsystem.h>

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
#define GL_DEPTH_COMPONENT16              0x81A5
#define GL_TEXTURE_COMPARE_FUNC           0x884D
#define GL_TEXTURE_COMPARE_MODE           0x884C
#define GL_COMPARE_R_TO_TEXTURE           0x884E
#define GL_CLAMP_TO_BORDER                0x812D

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
typedef void (__stdcall *PFNGLBUFFERDATAPROC) (GLenum t, size_t s, const GLvoid *d, GLenum u);
typedef void (__stdcall *PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void (__stdcall *PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void (__stdcall *PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void (__stdcall *PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void (__stdcall *PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);
typedef void (__stdcall *PFNGLACTIVETEXTUREPROC) (GLenum texture);

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

struct ObjectData 
{
	float* vertexmap;
	float* uvmap;
	float* normalmap;
	int trianglecount;
};

struct BufferData
{
	unsigned int vertexbuffer;
	unsigned int uvbuffer;
	unsigned int normalbuffer;
	int shader;
	unsigned int albedo;
	unsigned int alpha;
	struct ObjectData data;
};
	
float CameraRotationMatrix[4][4], ViewMatrix[4][4], ProjectionViewMatrix[4][4], MVP[4][4], iMouse[2] = {0.0f,0.0f};
float LightRotationMatrix[4][4], LightViewMatrix[4][4], LightMVP[4][4];
unsigned int framebuffer, depthmap;

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
}
	
static const char* DepthMapVertexShader = \
	"#version 460 \n"	
	"layout (location=0) in vec3 vertex;"
	"layout (location=1) in vec2 uv;"
	"out vec2 UV;"
	"uniform mat4 MVP;"	
	"uniform vec3 pivot;"
	"uniform vec3 rotation;"
	"uniform vec3 scale;"
	"mat3 rotationX( float x) {return mat3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));}"
	"mat3 rotationY( float y) {return mat3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));}"
	"mat3 rotationZ( float z) {return mat3(cos(z),sin(z),0.0,-sin(z),cos(z),0.0,0.0,0.0,1.0);}"	
	"void main()"
	"{"	
		"gl_Position = MVP * vec4((rotationX(rotation.x) * rotationY(rotation.y) * rotationZ(rotation.z) * vertex ) * scale + pivot, 1.0);"
		"UV = uv;"
	"}";
	
static const char* DepthMapFragmentShader = \
	"#version 460 \n"
	"out float fragColor;"
	"in vec2 UV;"
	"uniform sampler2D _MainTex;"
	"uniform sampler2D _AlphaMap;"
	"void main()"
	"{"	
		"if((texture(_AlphaMap,UV).r - 0.5) < 0) discard;"
		"fragColor = gl_FragCoord.z;"
	"}";

static const char* ShadowVertexShader = \
	"#version 460 \n"	
	"layout (location=0) in vec3 vertex;"
	"layout (location=1) in vec2 uv;"
	"layout (location=2) in vec3 normal;"
	"out vec2 UV;"
	"out vec3 NORMAL;"
	"out vec4 ShadowCoord;"
	"uniform mat4 MVP;"
	"uniform mat4 LightMVP;"
	"uniform vec3 UVscale;"
	"uniform vec3 UVoffset;"	
	"uniform vec3 pivot;"
	"uniform vec3 rotation;"
	"uniform vec3 scale;"
	"mat3 rotationX( float x) {return mat3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));}"
	"mat3 rotationY( float y) {return mat3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));}"
	"mat3 rotationZ( float z) {return mat3(cos(z),sin(z),0.0,-sin(z),cos(z),0.0,0.0,0.0,1.0);}"	
	"void main()"
	"{"	
		"gl_Position = MVP * vec4((rotationX(rotation.x) * rotationY(rotation.y) * rotationZ(rotation.z) * vertex ) * scale + pivot, 1.0);"
		"UV = uv * UVscale.xy + UVoffset.xy;"
		"NORMAL = normalize( (rotationX(rotation.x) * rotationY(rotation.y) * rotationZ(rotation.z) * normal) );"
		"ShadowCoord = LightMVP * vec4((rotationX(rotation.x) * rotationY(rotation.y) * rotationZ(rotation.z) * vertex ) * scale + pivot, 1.0);"
	"}";
	
static const char* ShadowFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"in vec2 UV;"
	"in vec3 NORMAL;"
	"in vec4 ShadowCoord;"
	"uniform sampler2D _MainTex;"
	"uniform sampler2D _AlphaMap;"
	"uniform sampler2D _ShadowMap;"	
	"void main()"
	"{"	
		"if((texture(_AlphaMap,UV).r - 0.5) < 0) discard;"
		"vec3 diffuse = max(dot(normalize(vec3(0,0,100)),NORMAL),0.0) * vec3(1,1,0) + vec3(0.5);"
		"vec3 projCoords = ShadowCoord.xyz / ShadowCoord.w;"
		"projCoords = projCoords * 0.5 + 0.5;"
		"float closestDepth = texture(_ShadowMap, projCoords.xy).r; "
		"float currentDepth = projCoords.z;"
		"float bias = 0.0005;"
		"float shadow = currentDepth-bias > closestDepth  ? 1.0 : 0.0;"
		"if(projCoords.z > 1.0) shadow = 0.0;"
		"fragColor = vec4(1.25-shadow) * texture(_MainTex,UV) * vec4(diffuse,1.0);"	
	"}";
	
static const char* GrassVertexShader = \
	"#version 460 \n"
	"uniform sampler2D _MainTex;"
	"uniform sampler2D _GrassSurface;"
	"uniform mat4 MVP;"
	"out vec3 worldPos;"
	"out vec2 UV;"
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"int q = int(floor(float(id) / 6.0));"
		"vec3 center = 256.0 * texelFetch(_MainTex, ivec2(int(mod(float(q),256.)),int(floor(float(q) / 256.0))), 0).rgb;"
		"center.y += 0.5;"
		"center*=vec3(0.2,0.2,0.2);"
		"if (mod(id,6)==0) "
		"{"
			"UV = vec2(0,0);"
			"worldPos = vec3(-0.5,-0.5,0) + center;"
			"gl_Position = vec4( MVP * vec4(worldPos, 1));"
		"}"
		"else if (mod(id,6)==1) "
		"{"
			"UV = vec2(1,0);"
			"worldPos = vec3(0.5,-0.5,0) + center;"
			"gl_Position = vec4( MVP *vec4(worldPos, 1));"
		"}"
		"else if (mod(id,6)==2) "
		"{"
			"UV = vec2(0,1);"
			"worldPos = vec3(-0.5,0.5,0) + center;"
			"gl_Position = vec4( MVP *vec4(worldPos, 1));"
		"}"
		"else if (mod(id,6)==3) "
		"{"
			"UV = vec2(1,0);"
			"worldPos = vec3(0.5,-0.5,0) + center;"
			"gl_Position = vec4( MVP *vec4(worldPos, 1));"
		"}"
		"else if (mod(id,6)==4) "
		"{"
			"UV = vec2(1,1);"
			"worldPos = vec3(0.5,0.5,0) + center;"
			"gl_Position = vec4( MVP *vec4(worldPos, 1));"
		"}"
		"else "
		"{"
			"UV = vec2(0,1);"
			"worldPos = vec3(-0.5,0.5,0) + center;"
			"gl_Position = vec4( MVP *vec4(worldPos, 1));"
		"}"		
	"}";
	
static const char* GrassFragmentShader = \
	"#version 460 \n"
	"out vec4 color;"
	"uniform sampler2D _MainTex;"
	"uniform sampler2D _GrassSurface;"
	"uniform sampler2D _GrassAlpha;"	
	"in vec2 UV;"
	"in vec3 worldPos;"
	"void main()"
	"{"
		"if((texture(_GrassAlpha,UV).r - 0.5) < 0) discard;"
		"vec3 dx = dFdxFine(worldPos);"
		"vec3 dy = dFdyFine(worldPos);"
		"vec3 normal = normalize(cross(dx,dy));"
		"vec3 diffuse = vec3(max(dot(normalize(vec3(0,0,-100)),normal),0.0)) + 0.2;"
		"color = texture(_GrassSurface,UV) * vec4(diffuse, 1.0);"
	"}";

static const char* LoadingScreenVertexShader = \
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
	
static const char* LoadingScreenFragmentShader = \
	"#version 450 \n"
	"out vec4 fragColor;"
	"in vec2 UV;"
	
	"float l(vec2 uv)"
	"{"
		"uv.y -= .2;"
		"return length(vec2(uv.x,max(0.,abs(uv.y)-.6)))+.4;"
	"}"

	"float o(vec2 uv)"
	"{"
		"return abs(length(vec2(uv.x,max(0.,abs(uv.y)-.15)))-.25)+.4;"
	"}"

	"float a(vec2 uv)"
	"{"
		"uv = -uv;"
		"float x = abs(length(vec2(max(0.,abs(uv.x)-.05),uv.y-.2))-.2)+.4;"
		"x = min(x,length(vec2(uv.x+.25,max(0.,abs(uv.y-.2)-.2)))+.4);"
		"return min(x,(uv.x<0.?uv.y<0.:atan(uv.x,uv.y+0.15)>2.)?abs(length(vec2(uv.x,max(0.,abs(uv.y)-.15)))-.25)+.4:length(vec2(uv.x-.22734,uv.y+.254))+.4);"
	"}"

	"float d(vec2 uv)"
	"{"
		"uv.x *= -1.;"
		"float x = abs(length(vec2(uv.x,max(0.,abs(uv.y)-.15)))-.25)+.4;"
		"uv.x += .25;"
		"uv.y -= .2;"
		"return min(x,length(vec2(uv.x,max(0.,abs(uv.y)-.6)))+.4);"
	"}"

	"float i(vec2 uv)" 
	"{"
		"return min(length(vec2(uv.x,max(0.,abs(uv.y)-.4)))+.4,length(vec2(uv.x,uv.y-.7))+.4);"
	"}"

	"float n(vec2 uv)"
	"{"
		"uv.y *= -1.;"
		"float x = length(vec2(abs(length(vec2(uv.x,max(0.0,-(.4-0.25)-uv.y) ))-0.25),max(0.,uv.y-.4))) +.4;"
		"uv.x+=.25;"
		"return min(x,length(vec2(uv.x,max(0.,abs(uv.y)-.4)))+.4);"
	"}"

	"float g(vec2 uv)"
	"{"
		"float x = abs(length(vec2(uv.x,max(0.,abs(uv.y)-.15)))-.25)+.4;"
		"return min(x,uv.x>0.||uv.y<-.65?length(vec2(abs(length(vec2(uv.x,max(0.0,-(.4+0.2)-uv.y) ))-.25),max(0.,uv.y-.4))) +.4:length(vec2(uv.x+0.25,uv.y+.65))+.4 );"
	"}"
	
	"void main ()"
	"{"
		"vec2 uv = 2.0* UV -1.0;" 
		"bool T[7];"
		"T[0] = l(uv*5.0 + vec2 (3.0, 0.0))<0.5;"
		"T[1] = o(uv*5.0 + vec2 (2.0, 0.0))<0.5;"
		"T[2] = a(uv*5.0 + vec2 (1.0, 0.0))<0.5;"
		"T[3] = d(uv*5.0 + vec2 (0.0, 0.0))<0.5;"
		"T[4] = i(uv*5.0 + vec2 (-1.0, 0.0))<0.5;"
		"T[5] = n(uv*5.0 + vec2 (-2.0, 0.0))<0.5;"
		"T[6] = g(uv*5.0 + vec2 (-3.0, 0.0))<0.5;"
		"if(T[0] ||  T[1] || T[2] ||  T[3] ||  T[4] ||  T[5] ||  T[6]) fragColor = vec4(1,1,1,1); else fragColor = vec4(0,0,0,1) ;"
	"}";	

static const char* SkyboxVertexShader = \
	"#version 460 \n"
	"out vec3 UV;"
	"const vec3 skybox[36] = "
	"{"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f, -1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3( 1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3(-1.0f, -1.0f, -1.0f),"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f,  1.0f,  1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3( 1.0f, -1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f, -1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3(-1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f, -1.0f,  1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3( 1.0f,  1.0f, -1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3( 1.0f,  1.0f,  1.0f),"
		"vec3(-1.0f,  1.0f,  1.0f),"
		"vec3(-1.0f,  1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3( 1.0f, -1.0f, -1.0f),"
		"vec3(-1.0f, -1.0f,  1.0f),"
		"vec3( 1.0f, -1.0f,  1.0f)"
	"};"
	"uniform mat4 MVP;"
	"uniform vec3 offset;"	
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"UV = skybox[id] ;"
		"gl_Position = MVP * vec4(skybox[id] * 500.0 + offset, 1.0);"
	"}";

static const char* SkyboxFragmentShader = \
	"#version 460 \n"
	"layout(location = 0) out vec4 fragColor;"	
	"in vec3 UV;"
	"uniform samplerCube skybox;"	

	"void main()"
	"{"	
		"fragColor = texture(skybox, UV);"
	"}";	

float CameraTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,1.8f,
	0.0f,0.0f,-1.0f,-8.0f,
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

float LightTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,20.0f,
	0.0f,1.0f,0.0f,7.0f,
	0.0f,0.0f,-1.0f,45.0f,
	0.0f,0.0f,0.0f,1.0f
};

float LightRotationYMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float LightRotationXMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float OrthographicMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

void OrthographicProjection (float left, float right, float bottom, float top, float cnear, float cfar, float k[4][4])
{
	k[0][0] = 2.0f / (right - left);
	k[0][1] = 0.0f;
	k[0][2] = 0.0f;
	k[0][3] = (-1.0f) * (right + left) / (right - left);
	k[1][0] = 0.0f;
	k[1][1] = 2.0f / (top - bottom);
	k[1][2] = 0.0f;
	k[1][3] = (-1.0f) * (top + bottom) / (top - bottom);
	k[2][0] = 0.0f;
	k[2][1] = 0.0f;
	k[2][2] = (-2.0f) / (cfar - cnear);
	k[2][3] = (-1.0f) * (cfar + cnear) / (cfar - cnear);
	k[3][0] = 0.0f;
	k[3][1] = 0.0f;
	k[3][2] = 0.0f;
	k[3][3] = 1.0f;
}

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

unsigned int LoadTexture(char *filename)
{
	int width, height; 
	char buffer[128]; 
	FILE *file = fopen(filename, "rb");
	fgets(buffer, sizeof(buffer), file);
	do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
	sscanf (buffer, "%d %d", &width, &height);
	do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
	int size = width * height * 4 * sizeof(GLubyte);
	GLubyte *Texels  = (GLubyte *)malloc(size);
	for (int i = 0; i < size; i++) 
	{
		Texels[i] = ((i % 4) < 3 ) ? (GLubyte) fgetc(file) : (GLubyte) 255 ;
	}
	fclose(file);
	unsigned int id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);	
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAX_ANISOTROPY,8);
	free(Texels);
	return id;	
}

char* Concat( char* s1, char* s2 )
{
	char* s3 = (char*) malloc(1+strlen(s1)+strlen(s2));
	strcpy(s3, s1);
	strcat(s3, s2);
	return s3;
}

unsigned int LoadCubemap(char *path)
{
	unsigned int textureID;
	glGenTextures(1, &textureID);
	glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
	for (unsigned int i = 0; i < 6; i++)
	{
		int width, height; 
		char buffer[128]; 
		char counter[5];
		itoa(i, counter, 10);
		FILE *file = fopen(Concat(path, Concat(counter,".ppm")), "rb");
		fgets(buffer, sizeof(buffer), file);
		do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
		sscanf (buffer, "%d %d", &width, &height);
		do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
		int size = width * height * 4 * sizeof(GLubyte);
		GLubyte *Texels  = (GLubyte *)malloc(size);
		for (int i = 0; i < size; i++) 
		{
			Texels[i] = ((i % 4) < 3 ) ? (GLubyte) fgetc(file) : (GLubyte) 255 ;
		}
		fclose(file);
		glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i , 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Texels);
		free(Texels);
	}
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	return textureID;
}

void LoadDepthBuffer ()
{	
	glGenFramebuffers(1, &framebuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glGenTextures(1, &depthmap);
	glBindTexture(GL_TEXTURE_2D, depthmap);
	glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT16, 2048, 2048, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
	glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthmap, 0);
	glDrawBuffer(GL_NONE); 		
}

unsigned int LoadGrassBuffer()
{
	int size = 256 * 256 * 4 * sizeof(GLubyte);
	GLubyte *Texels  = (GLubyte *)malloc(size);	
	for (int i=0; i<(256 * 256); i++)
	{
		Texels[i*4+0] =  fmodf((float)i, 256.0f);
		Texels[i*4+1] =  0;  
		Texels[i*4+2] =  floorf((float)i / 256.0f);
		Texels[i*4+3] =  255;
	}	
	unsigned int id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);	
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,256,256,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	free(Texels);
	return id;	
}

struct ObjectData LoadGeometry(char* path)
{
	FILE* file = fopen(path, "r");
	char line[5000];
	int vertexcount = 0;
	float* vertices = (float*)malloc(sizeof(float));
	int normalcount = 0;
	float* normals = (float*)malloc(sizeof(float));
	int uvcount = 0;
	float* uvs = (float*)malloc(sizeof(float));
	int indicescount = 0;
	int* indices = (int*)malloc(sizeof(int));
	while (1)
	{
		if (fgets(line,200, file) == NULL) break;
		char *ptr = strtok(line, " ");
		if(strcmp(&ptr[0], "v") == 0)
		{
			ptr = strtok(NULL," ");
			while(ptr != NULL)
			{
				vertices = realloc(vertices,sizeof(float)*(vertexcount+1));
				vertices[vertexcount] = atof(ptr);
				vertexcount++;
				ptr = strtok(NULL, " ");
			}
		}
		else if(strcmp(&ptr[0], "vn") == 0)
		{
			ptr = strtok(NULL," ");
			while(ptr != NULL)
			{
				normals = realloc(normals,sizeof(float)*(normalcount+1));
				normals[normalcount] = atof(ptr);
				normalcount++;
				ptr = strtok(NULL, " ");
			}
		}
		else if(strcmp(&ptr[0], "vt") == 0)
		{
			ptr = strtok(NULL," ");
			while(ptr != NULL)
			{
				uvs = realloc(uvs,sizeof(float)*(uvcount+1));
				uvs[uvcount] = atof(ptr);
				uvcount++;
				ptr = strtok(NULL, " ");
			}
		}
		else
		{
			ptr = strtok(NULL," /");
			while(ptr != NULL)
			{
				indices = realloc(indices,sizeof(int)*(indicescount+1));
				indices[indicescount] = atoi(ptr);
				indicescount++;	
				ptr = strtok(NULL, " /");
			}
		}
	}
	fclose(file);
	struct ObjectData data;
	data.trianglecount = indicescount/3;
	data.vertexmap = (float*)malloc(data.trianglecount * 3 * sizeof(float));
	data.normalmap = (float*)malloc(data.trianglecount * 3 * sizeof(float));
	data.uvmap = (float*)malloc(data.trianglecount * 2 * sizeof(float));
	for (int i=0; i<data.trianglecount; i++)
	{
		// reversed face winding
		data.vertexmap[i*3+2] = vertices[indices[i*3]*3-3+0];
		data.vertexmap[i*3+1] = vertices[indices[i*3]*3-3+1];
		data.vertexmap[i*3+0] = vertices[indices[i*3]*3-3+2];
		data.uvmap[i*2+0] = uvs[indices[i*3+1]*2-2+0];
		data.uvmap[i*2+1] = uvs[indices[i*3+1]*2-2+1];
		data.normalmap[i*3+2] = normals[indices[i*3+2]*3-3+0];
		data.normalmap[i*3+1] = normals[indices[i*3+2]*3-3+1];
		data.normalmap[i*3+0] = normals[indices[i*3+2]*3-3+2];
	}
	free(vertices);
	free(uvs);
	free(normals);
	free(indices);
	return data;
}

struct BufferData LoadObject(char* geometry, char* diffusemap, char* alphamap, const char* vshader, const char* pshader)
{
	struct BufferData bufferdata;
	unsigned int VertexArrayID;
	struct ObjectData data = LoadGeometry(geometry);
	bufferdata.albedo = LoadTexture(diffusemap);
	bufferdata.alpha = LoadTexture(alphamap);
	glGenVertexArrays (1, &VertexArrayID);		
	glBindVertexArray (VertexArrayID);	
	glGenBuffers(1, &bufferdata.vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.vertexbuffer);
	glBufferData(GL_ARRAY_BUFFER, data.trianglecount * 3* sizeof(float), data.vertexmap, GL_STATIC_DRAW);
	glGenBuffers(1, &bufferdata.uvbuffer);	
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, data.trianglecount * 2 * sizeof(float), data.uvmap, GL_STATIC_DRAW);
	glGenBuffers(1, &bufferdata.normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, data.trianglecount * 3 * sizeof(float), data.normalmap, GL_STATIC_DRAW);
	bufferdata.shader = LoadShaders(vshader, pshader);
	bufferdata.data = data;	
	return bufferdata;
}

void RenderDepth (struct BufferData bufferdata, float lightmvpt[4][4], float px, float py, float pz, float rx, float ry, float rz, float sx, float sy, float sz)
{	
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, bufferdata.albedo);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, bufferdata.alpha);
	glUseProgram(bufferdata.shader);
	glUniform1i(glGetUniformLocation(bufferdata.shader, "_MainTex"), 0);
	glUniform1i(glGetUniformLocation(bufferdata.shader, "_AlphaMap"), 1);
	glUniform3f(glGetUniformLocation(bufferdata.shader,"rotation"), rx, ry, rz);
	glUniform3f(glGetUniformLocation(bufferdata.shader,"pivot"), px, py, pz);
	glUniform3f(glGetUniformLocation(bufferdata.shader,"scale"), sx, sy, sz);	
	glUniformMatrix4fv(glGetUniformLocation(bufferdata.shader,"MVP"), 1, GL_FALSE, &lightmvpt[0][0]);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.vertexbuffer);
	glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );
	glEnableVertexAttribArray(1);	
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.uvbuffer);		
	glVertexAttribPointer(1,2, GL_FLOAT, GL_FALSE, 0,(void*)0 );
	glDrawArrays(GL_TRIANGLES, 0, bufferdata.data.trianglecount);
	glActiveTexture(GL_TEXTURE0 + 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);	
}

void RenderObjectWithShadow (unsigned int shadowmap, struct BufferData bufferdata, float mvpt[4][4], float lightmvpt[4][4], float px, float py, float pz, float rx, float ry, float rz, float sx, float sy, float sz, float ox, float oy, float oz, float ux, float uy, float uz)
{	
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, bufferdata.albedo);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, bufferdata.alpha);
	glActiveTexture(GL_TEXTURE0 + 2);
	glBindTexture(GL_TEXTURE_2D, shadowmap);		
	glUseProgram(bufferdata.shader);
	glUniform1i(glGetUniformLocation(bufferdata.shader, "_MainTex"), 0);
	glUniform1i(glGetUniformLocation(bufferdata.shader, "_AlphaMap"), 1);
	glUniform1i(glGetUniformLocation(bufferdata.shader, "_ShadowMap"), 2);	
	glUniform3f(glGetUniformLocation(bufferdata.shader,"rotation"), rx, ry, rz);	
	glUniform3f(glGetUniformLocation(bufferdata.shader,"UVscale"), ux, uy, uz);
	glUniform3f(glGetUniformLocation(bufferdata.shader,"UVoffset"), ox, oy, oz);	
	glUniform3f(glGetUniformLocation(bufferdata.shader,"pivot"), px, py, pz);
	glUniform3f(glGetUniformLocation(bufferdata.shader,"scale"), sx, sy, sz);	
	glUniformMatrix4fv(glGetUniformLocation(bufferdata.shader,"MVP"), 1, GL_FALSE, &mvpt[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(bufferdata.shader,"LightMVP"), 1, GL_FALSE, &lightmvpt[0][0]);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.vertexbuffer);
	glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );
	glEnableVertexAttribArray(1);	
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.uvbuffer);		
	glVertexAttribPointer(1,2, GL_FLOAT, GL_FALSE, 0,(void*)0 );
	glEnableVertexAttribArray(2);	
	glBindBuffer(GL_ARRAY_BUFFER, bufferdata.normalbuffer);		
	glVertexAttribPointer(2,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );
	glDrawArrays(GL_TRIANGLES, 0, bufferdata.data.trianglecount);
	glActiveTexture(GL_TEXTURE0 + 0);
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);	
}

void RenderLoadingScreen (HDC handle)
{
	glUseProgram(LoadShaders(LoadingScreenVertexShader, LoadingScreenFragmentShader));
	glDrawArrays(GL_TRIANGLES, 0, 6);
	wglSwapLayerBuffers(handle, WGL_SWAP_MAIN_PLANE);	
}

void RenderSkybox (unsigned int shader, unsigned int cubemap, float mvp[4][4])
{
	glUseProgram(shader);
	glUniform3f(glGetUniformLocation(shader,"offset"), CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]);
	glUniformMatrix4fv(glGetUniformLocation(shader,"MVP"), 1, GL_FALSE, &mvp[0][0]);
	glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
	glDrawArrays(GL_TRIANGLES, 0, 36);	
}

void RenderGrass (unsigned int buffer, unsigned int surface, unsigned int alpha, unsigned int shader, float mvp[4][4])
{
	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, buffer);
	glActiveTexture(GL_TEXTURE0 + 1);
	glBindTexture(GL_TEXTURE_2D, surface);
	glActiveTexture(GL_TEXTURE0 + 2);
	glBindTexture(GL_TEXTURE_2D, alpha);		
	glUseProgram(shader);
	glUniform1i(glGetUniformLocation(shader, "_MainTex"), 0);
	glUniform1i(glGetUniformLocation(shader, "_GrassSurface"), 1);
	glUniform1i(glGetUniformLocation(shader, "_GrassAlpha"), 2);
	glUniformMatrix4fv(glGetUniformLocation(shader, "MVP"), 1, GL_FALSE, &mvp[0][0]);
	glDrawArrays(GL_TRIANGLES, 0, 6 * 256 * 256);
	glActiveTexture(GL_TEXTURE0 + 0); 	
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
	if (deltaX>0.0f) iMouse[0]+=3.5f; 
	if (deltaX<0.0f) iMouse[0]-=3.5f; 
	if (deltaZ>0.0f) iMouse[1]+=3.5f; 
	if (deltaZ<0.0f) iMouse[1]-=3.5f; 
	CameraRotationXMatrix[1][1] = cos(deg2rad(iMouse[0]));
	CameraRotationXMatrix[1][2] = (-1.0f)*sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][1] = sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][2] = cos(deg2rad(iMouse[0]));
	CameraRotationYMatrix[0][0] = cos(deg2rad(iMouse[1]));
	CameraRotationYMatrix[0][2] = sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][0] = (-1.0f)*sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][2] = cos(deg2rad(iMouse[1]));
}

float Box (float p[3], float c[3], float s[3])
{
	float x = fmaxf(p[0] - c[0]- s[0], c[0] - p[0] - s[0]);
	float y = fmaxf(p[1] - c[1]- s[1], c[1] - p[1] - s[1]);   
	float z = fmaxf(p[2] - c[2]- s[2], c[2] - p[2]- s[2]);
	float d = x;
	d = fmaxf(d,y);
	d = fmaxf(d,z);
	return d;
}

bool Collision(float p[3])   //returns true if collide
{
	float c1[3] = {5.0f,1.0f,0.0f};
	float s1[3] = {1.0f,1.0f,1.0f};
	float t1 = Box(p, c1, s1);
	float c2[3] = {0.0f,0.0f,0.0f};
	float s2[3] = {100.0f,0.02f,100.0f};
	float t2 = Box(p, c2, s2);   //ground plane collider
	float c3[3] = {-20.0f,5.0f,0.0f};
	float s3[3] = {1.1f,5.1f,5.1f};
	float t3 = Box(p, c3, s3);	
	float c4[3] = {0.0f,0.0f,0.0f};
	float s4[3] = {2.0f,3.0f,4.0f};
	float t4 = Box(p, c4, s4);   //wreck collider	
	return (t2 <= 0.0f || t4 <= 0.0f);
}

void KeyboardMovement()
{
	float dx = 0.0f;
	float dz = 0.0f;
	float cx = 0.0f;
	float cz = 0.0f;
	float p[3] = {CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]};
	if (GetAsyncKeyState(0x57) ) cz =  50.0f;
	if (GetAsyncKeyState(0x53) ) cz = -50.0f;
	if (GetAsyncKeyState(0x44) ) cx =  50.0f;
	if (GetAsyncKeyState(0x41) ) cx = -50.0f;	
	p[0] += (-cz * ViewMatrix[2][0] + cx * ViewMatrix[0][0]) * 0.01f;
	p[1] += (-cz * ViewMatrix[2][1] + cx * ViewMatrix[1][0]) * 0.01f;
	p[2] += (-cz * ViewMatrix[2][2] + cx * ViewMatrix[2][0]) * 0.01f;
	bool allow = !Collision(p);
	if (GetAsyncKeyState(0x57) && allow) dz =  50.0f;
	if (GetAsyncKeyState(0x53) && allow) dz = -50.0f ;
	if (GetAsyncKeyState(0x44) && allow) dx =  50.0f;
	if (GetAsyncKeyState(0x41) && allow) dx = -50.0f ;
	CameraTranslationMatrix[0][3] += (-dz * ViewMatrix[2][0] + dx * ViewMatrix[0][0]) * 0.001f;
	CameraTranslationMatrix[1][3] += (-dz * ViewMatrix[2][1] + dx * ViewMatrix[1][0]) * 0.001f;  //comment to enable FPP
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
	RenderLoadingScreen (hdc);
	struct BufferData terrain = LoadObject("geometry//terrain3.obj","textures\\forestground3.ppm","textures\\alphamap.ppm",DepthMapVertexShader,DepthMapFragmentShader);
	struct BufferData branches = LoadObject("geometry//tree//branches.obj","textures\\tree.ppm","textures\\treealpha.ppm",DepthMapVertexShader,DepthMapFragmentShader);
	struct BufferData cap = LoadObject("geometry//tree//cap.obj","textures\\tree.ppm","textures\\treealpha.ppm",DepthMapVertexShader,DepthMapFragmentShader);	
	struct BufferData bark = LoadObject("geometry//tree//bark.obj","textures\\tree.ppm","textures\\treealpha.ppm",DepthMapVertexShader,DepthMapFragmentShader);	
	struct BufferData leaves1 = LoadObject("geometry//tree//leaves1.obj","textures\\tree.ppm","textures\\treealpha.ppm",DepthMapVertexShader,DepthMapFragmentShader);	
	struct BufferData leaves2 = LoadObject("geometry//tree//leaves2.obj","textures\\tree.ppm","textures\\treealpha.ppm",DepthMapVertexShader,DepthMapFragmentShader);
	struct BufferData shadowterrain = LoadObject("geometry//terrain3.obj","textures\\forestground3.ppm","textures\\alphamap.ppm",ShadowVertexShader,ShadowFragmentShader);
	struct BufferData shadowbranches = LoadObject("geometry//tree//branches.obj","textures\\tree.ppm","textures\\treealpha.ppm",ShadowVertexShader,ShadowFragmentShader);
	struct BufferData shadowcap = LoadObject("geometry//tree//cap.obj","textures\\tree.ppm","textures\\treealpha.ppm",ShadowVertexShader,ShadowFragmentShader);	
	struct BufferData shadowbark = LoadObject("geometry//tree//bark.obj","textures\\tree.ppm","textures\\treealpha.ppm",ShadowVertexShader,ShadowFragmentShader);	
	struct BufferData shadowleaves1 = LoadObject("geometry//tree//leaves1.obj","textures\\tree.ppm","textures\\treealpha.ppm",ShadowVertexShader,ShadowFragmentShader);	
	struct BufferData shadowleaves2 = LoadObject("geometry//tree//leaves2.obj","textures\\tree.ppm","textures\\treealpha.ppm",ShadowVertexShader,ShadowFragmentShader);		
	int GrassShader = LoadShaders(GrassVertexShader, GrassFragmentShader);
	int GrassBufferID = LoadGrassBuffer();
	unsigned int GrassTexture = LoadTexture("textures\\grass.ppm");
	unsigned int GrassAlpha = LoadTexture("textures\\grassalpha.ppm"); 
	int SK = LoadShaders(SkyboxVertexShader, SkyboxFragmentShader);		
	unsigned int cubemap = LoadCubemap("textures\\"); 
	ProjectionMatrix[0][0] = ((1.0f/tan(deg2rad(FieldOfView/2.0f)))/(ScreenWidth/ScreenHeight));
	ProjectionMatrix[1][1] = (1.0f/tan(deg2rad(FieldOfView/2.0f)));
	ProjectionMatrix[2][2] = (-1.0f)* (FarClip+NearClip)/(FarClip-NearClip);
	ProjectionMatrix[2][3] = (-1.0f)*(2.0f*FarClip*NearClip)/(FarClip-NearClip);
	LightRotationXMatrix[1][1] = cos(deg2rad(-35.0f));
	LightRotationXMatrix[1][2] = (-1.0f)*sin(deg2rad(-35.0f));
	LightRotationXMatrix[2][1] = sin(deg2rad(-35.0f));
	LightRotationXMatrix[2][2] = cos(deg2rad(-35.0f));
	LightRotationYMatrix[0][0] = cos(deg2rad(180.0f));
	LightRotationYMatrix[0][2] = sin(deg2rad(180.0f));
	LightRotationYMatrix[2][0] = (-1.0f)*sin(deg2rad(180.0f));
	LightRotationYMatrix[2][2] = cos(deg2rad(180.0f));
	OrthographicProjection(-30.0f,30.0f,-30.0f,30.0f,0.1f,50.0f, OrthographicMatrix);	
	Mul(LightRotationYMatrix,LightRotationXMatrix,LightRotationMatrix);
	Mul(LightTranslationMatrix,LightRotationMatrix,LightViewMatrix);
	Inverse(LightViewMatrix,LightViewMatrix);
	Mul(OrthographicMatrix,LightViewMatrix,LightMVP);	
	float LightMVPT[4][4] = 
	{
		LightMVP[0][0], LightMVP[1][0], LightMVP[2][0], LightMVP[3][0],
		LightMVP[0][1], LightMVP[1][1], LightMVP[2][1], LightMVP[3][1],
		LightMVP[0][2], LightMVP[1][2], LightMVP[2][2], LightMVP[3][2],
		LightMVP[0][3], LightMVP[1][3], LightMVP[2][3], LightMVP[3][3]
	};	
	sndPlaySound("audio//wind.wav",SND_LOOP | SND_ASYNC);
	LoadDepthBuffer ();
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
		// ---------------------------------------------------------------------------------------------------------------------
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );
		glViewport(0, 0, 2048, 2048);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
		RenderDepth (terrain, LightMVPT, -128.0f,0.0f,128.0f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);		
		for (int i=0; i<10; i++)
		{
			for (int j=0; j<10; j++)
			{
				RenderDepth (branches, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
				RenderDepth (cap, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
				RenderDepth (bark, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
				RenderDepth (leaves1, LightMVPT,4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
				RenderDepth (leaves2, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f);
			}
		}
		// ---------------------------------------------------------------------------------------------------------------------
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, ScreenWidth, ScreenHeight);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);
		RenderSkybox (SK, cubemap, MVPT);
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );
		RenderObjectWithShadow (depthmap, shadowterrain, MVPT, LightMVPT,-128.0f,0.0f,128.0f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f, 0.0f,0.0f,0.0f, 30.0f,30.0f,30.0f);
		for (int i=0; i<10; i++)
		{
			for (int j=0; j<10; j++)
			{
				RenderObjectWithShadow (depthmap, shadowbranches, MVPT, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);
				RenderObjectWithShadow (depthmap, shadowcap, MVPT, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);
				RenderObjectWithShadow (depthmap, shadowbark, MVPT, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);
				RenderObjectWithShadow (depthmap, shadowleaves1, MVPT, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);
				RenderObjectWithShadow (depthmap, shadowleaves2, MVPT, LightMVPT, 4*i,0.0f,4*j, 0.0f,0.0f,0.0f, 0.2f,0.2f,0.2f, 0.0f,0.0f,0.0f, 1.0f,1.0f,1.0f);
			}
		}
		RenderGrass (GrassBufferID, GrassTexture, GrassAlpha, GrassShader, MVPT);
		// ---------------------------------------------------------------------------------------------------------------------
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	}
	return 0;
}