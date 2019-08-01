// Author: Przemyslaw Zaworski
// cl demo1.c opengl32.lib user32.lib gdi32.lib winmm.lib
/*
	Features:
	- collision detection (three box colliders)
	- skybox rendering
	- keyboard and mouse movement
	- WAV audio loader
	- PPM texture loader
	- vignette post processing
*/

#include <windows.h>
#include <GL/gl.h>
#include <math.h>
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
}

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

static const char* QuadVertexShader = \
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

static const char* QuadFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"	
	"in vec2 UV;"
	"uniform sampler2D _MainTex;"	

	"void main()"
	"{"	
		"fragColor = texture(_MainTex,UV * 20);"
	"}";

static const char* SphereVertexShader = \
	"#version 460 \n"	
	"uniform mat4 MVP;"
	"out vec3 worldPos;"
	"out vec3 normal;"

	"void main()"
	"{"	
		"float f = gl_VertexID;"
		"float v = f - 6.0 * floor(f/6.0);"
		"f = (f - v) / 6.;"
		"float a = f - 256.0 * floor(f/256.0);"
		"f = (f-a) / 256.;"
		"float b = f-64.;"
		"a += (v - 2.0 * floor(v/2.0));"
		"b += v==2. || v>=4. ? 1.0 : 0.0;"
		"a = a/256.*6.28318;"
		"b = b/256.*6.28318;"
		"vec3 p = vec3(cos(a)*cos(b), sin(b), sin(a)*cos(b));"
		"normal = normalize(p);"
		"vec3 t = vec3(5.0, 1.0, 0.0);"
		"p += t;"
		"worldPos = p;"
		"gl_Position = MVP * vec4(p, 1);"
	"}";

static const char* SphereFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"in vec3 worldPos;"
	"in vec3 normal;"
	"uniform vec3 _WorldSpaceCameraPos;"
	"uniform samplerCube skybox;"
	
	"void main()"
	"{"	
		"vec3 I = normalize(worldPos - _WorldSpaceCameraPos);"
		"vec3 R = reflect(I, normal);"
		"fragColor = vec4(texture(skybox, R).rgb, 1.0);"	
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

static const char* CubeVertexShader = \
	"#version 460 \n"
	"out vec2 UV;"
	"const vec3 vertices[36] = "
	"{"
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f, -0.5f,  0.5f),"
		"vec3(-0.5f,  0.5f,  0.5f)," 
		"vec3( 0.5f,  0.5f, -0.5f)," 
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f,  0.5f, -0.5f),"
		"vec3( 0.5f, -0.5f,  0.5f),"
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3( 0.5f, -0.5f, -0.5f),"
		"vec3( 0.5f,  0.5f, -0.5f),"
		"vec3( 0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f,  0.5f,  0.5f),"
		"vec3(-0.5f,  0.5f, -0.5f),"
		"vec3( 0.5f, -0.5f,  0.5f),"
		"vec3(-0.5f, -0.5f,  0.5f),"
		"vec3(-0.5f, -0.5f, -0.5f),"
		"vec3(-0.5f,  0.5f,  0.5f),"
		"vec3(-0.5f, -0.5f,  0.5f),"
		"vec3( 0.5f, -0.5f,  0.5f),"
		"vec3( 0.5f,  0.5f,  0.5f),"
		"vec3( 0.5f, -0.5f, -0.5f),"
		"vec3( 0.5f,  0.5f, -0.5f),"
		"vec3( 0.5f, -0.5f, -0.5f),"
		"vec3( 0.5f,  0.5f,  0.5f),"
		"vec3( 0.5f, -0.5f,  0.5f),"
		"vec3( 0.5f,  0.5f,  0.5f),"
		"vec3( 0.5f,  0.5f, -0.5f),"
		"vec3(-0.5f,  0.5f, -0.5f),"
		"vec3( 0.5f,  0.5f,  0.5f),"
		"vec3(-0.5f,  0.5f, -0.5f),"
		"vec3(-0.5f,  0.5f,  0.5f),"
		"vec3( 0.5f,  0.5f,  0.5f),"
		"vec3(-0.5f,  0.5f,  0.5f),"
		"vec3( 0.5f, -0.5f,  0.5f)"
	"};"

	"const vec2 uvs[36] = "
	"{"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(0.1f, 0.5f),"
		"vec2(0.0f, 0.0f),"
		"vec2(0.0f, 0.5f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(0.1f, 0.5f),"
		"vec2(0.1f, 0.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 0.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(0.0f, 1.0f),"
		"vec2(1.0f, 1.0f),"
		"vec2(0.0f, 0.0f)"	
	"};"
	"uniform mat4 MVP;"
	
	"void main()"
	"{"	
		"vec3 offset = vec3(-20.0,5.0,0.0);"
		"uint id = gl_VertexID;"
		"UV = uvs[id] ;"
		"gl_Position = MVP * vec4(vertices[id] * vec3(2.0,10.0,10.0) + offset, 1.0);"
	"}";

static const char* CubeFragmentShader = \
	"#version 460 \n"
	"out vec4 fragColor;"	
	"in vec2 UV;"
	"uniform sampler2D _MainTex;"	

	"void main()"
	"{"	
		"fragColor = texture(_MainTex, UV);"
	"}";	

float CameraTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,1.0f,
	0.0f,0.0f,-1.0f,-5.0f,
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
	float t2 = Box(p, c2, s2);
	float c3[3] = {-20.0f,5.0f,0.0f};
	float s3[3] = {1.1f,5.1f,5.1f};
	float t3 = Box(p, c3, s3);	
	return (t1 <= 0.0f || t2 <= 0.0f || t3 <= 0.0f);
}

void KeyboardMovement()
{
	float dx = 0.0f;
	float dz = 0.0f;
	float cx = 0.0f;
	float cz = 0.0f;
	float p[3] = {CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]};
	if (GetAsyncKeyState(0x57) ) cz =  2.0f;
	if (GetAsyncKeyState(0x53) ) cz = -2.0f;
	if (GetAsyncKeyState(0x44) ) cx =  2.0f;
	if (GetAsyncKeyState(0x41) ) cx = -2.0f;	
	p[0] += (-cz * ViewMatrix[2][0] + cx * ViewMatrix[0][0]) * 0.01f;
	p[1] += (-cz * ViewMatrix[2][1] + cx * ViewMatrix[1][0]) * 0.01f;
	p[2] += (-cz * ViewMatrix[2][2] + cx * ViewMatrix[2][0]) * 0.01f;
	bool allow = !Collision(p);
	if (GetAsyncKeyState(0x57) && allow) dz =  2.0f;
	if (GetAsyncKeyState(0x53) && allow) dz = -2.0f ;
	if (GetAsyncKeyState(0x44) && allow) dx =  2.0f;
	if (GetAsyncKeyState(0x41) && allow) dx = -2.0f ;
	CameraTranslationMatrix[0][3] += (-dz * ViewMatrix[2][0] + dx * ViewMatrix[0][0]) * 0.001f;
	//CameraTranslationMatrix[1][3] += (-dz * ViewMatrix[2][1] + dx * ViewMatrix[1][0]) * 0.001f;
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
	int SQ = LoadShaders(QuadVertexShader, QuadFragmentShader);
	int SP = LoadShaders(SphereVertexShader, SphereFragmentShader);
	int SK = LoadShaders(SkyboxVertexShader, SkyboxFragmentShader);	
	int SC = LoadShaders(CubeVertexShader, CubeFragmentShader);
	int SV = LoadShaders(VignetteVertexShader, VignetteFragmentShader);	
	unsigned int gravel = LoadTexture("textures\\gravel.ppm");
	unsigned int plaster = LoadTexture("textures\\plaster.ppm");	
	unsigned int cubemap = LoadCubemap("textures\\"); 
	ProjectionMatrix[0][0] = ((1.0f/tan(deg2rad(FieldOfView/2.0f)))/(ScreenWidth/ScreenHeight));
	ProjectionMatrix[1][1] = (1.0f/tan(deg2rad(FieldOfView/2.0f)));
	ProjectionMatrix[2][2] = (-1.0f)* (FarClip+NearClip)/(FarClip-NearClip);
	ProjectionMatrix[2][3] = (-1.0f)*(2.0f*FarClip*NearClip)/(FarClip-NearClip);
	sndPlaySound("audio//wind.wav",SND_LOOP | SND_ASYNC);
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
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );
		glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
		glViewport(0, 0, ScreenWidth, ScreenHeight);	
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);	
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);	
		glUseProgram(SK);
		glUniform3f(glGetUniformLocation(SK,"offset"), CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]);
		glUniformMatrix4fv(glGetUniformLocation(SK,"MVP"), 1, GL_FALSE, &MVPT[0][0]);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
		glDrawArrays(GL_TRIANGLES, 0, 36);	
		glEnable( GL_DEPTH_TEST );
		glDepthMask( GL_TRUE );	
		glBindTexture(GL_TEXTURE_2D, gravel);
		glUseProgram(SQ);
		glUniformMatrix4fv(glGetUniformLocation(SQ,"MVP"), 1, GL_FALSE, &MVPT[0][0]);	
		glDrawArrays(GL_TRIANGLES, 0, 6);	
		glUseProgram(SP);
		glUniform3f(glGetUniformLocation(SP,"_WorldSpaceCameraPos"), CameraTranslationMatrix[0][3], CameraTranslationMatrix[1][3], CameraTranslationMatrix[2][3]);
		glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);	
		glUniformMatrix4fv(glGetUniformLocation(SP,"MVP"), 1, GL_FALSE, &MVPT[0][0]);	
		glDrawArrays(GL_TRIANGLES, 0, 196608);
		glBindTexture(GL_TEXTURE_2D, plaster);
		glUseProgram(SC);	
		glUniformMatrix4fv(glGetUniformLocation(SP,"MVP"), 1, GL_FALSE, &MVPT[0][0]);	
		glDrawArrays(GL_TRIANGLES, 0, 36);	
		glDepthMask(GL_FALSE);
		glDisable(GL_DEPTH_TEST);	
		glBindTexture(GL_TEXTURE_2D, colormap);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, ScreenWidth, ScreenHeight);	
		glUseProgram(SV);
		glUniform1i(glGetUniformLocation(SV, "_MainTex"), 0);	
		glDrawArrays(GL_TRIANGLES, 0, 6);	
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	}
	return 0;
}