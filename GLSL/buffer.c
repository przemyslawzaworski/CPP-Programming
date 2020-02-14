// cl buffer.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>

#define GL_SHADER_STORAGE_BUFFER          0x90D2
#define GL_READ_ONLY                      0x88B8
#define GL_DYNAMIC_DRAW                   0x88E8
#define GL_SHADER_STORAGE_BARRIER_BIT     0x00002000
#define GL_COMPUTE_SHADER                 0x91B9

typedef GLuint(__stdcall *PFNGLCREATEPROGRAMPROC)();
typedef GLuint(__stdcall *PFNGLCREATESHADERPROC)(GLenum t);
typedef void(__stdcall *PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(__stdcall *PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(__stdcall *PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(__stdcall *PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(__stdcall *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void (__stdcall *PFNGLDISPATCHCOMPUTEPROC) (GLuint x, GLuint y, GLuint z);
typedef void (__stdcall *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (__stdcall *PFNGLBINDBUFFERBASEPROC) (GLenum target, GLuint index, GLuint buffer);
typedef void (__stdcall *PFNGLBUFFERDATAPROC) (GLenum target, signed long int size, const void *data, GLenum usage);
typedef void (__stdcall *PFNGLMEMORYBARRIERPROC) (GLbitfield barriers);
typedef void *(__stdcall *PFNGLMAPBUFFERPROC) (GLenum target, GLenum access);
typedef GLboolean (__stdcall *PFNGLUNMAPBUFFERPROC) (GLenum target);
typedef void (__stdcall *PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);

PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLDISPATCHCOMPUTEPROC glDispatchCompute;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLBINDBUFFERBASEPROC glBindBufferBase;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLMEMORYBARRIERPROC glMemoryBarrier;
PFNGLMAPBUFFERPROC glMapBuffer;
PFNGLUNMAPBUFFERPROC glUnmapBuffer;
PFNGLBINDBUFFERPROC glBindBuffer;

void glInit()
{
	glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
	glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
	glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
	glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
	glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
	glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
	glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
	glDispatchCompute = (PFNGLDISPATCHCOMPUTEPROC)wglGetProcAddress("glDispatchCompute");
	glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers");
	glBindBufferBase = (PFNGLBINDBUFFERBASEPROC)wglGetProcAddress("glBindBufferBase");
	glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
	glMemoryBarrier = (PFNGLMEMORYBARRIERPROC)wglGetProcAddress("glMemoryBarrier");
	glMapBuffer = (PFNGLMAPBUFFERPROC)wglGetProcAddress("glMapBuffer");
	glUnmapBuffer = (PFNGLUNMAPBUFFERPROC)wglGetProcAddress("glUnmapBuffer");
	glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
}

static const char* ComputeShader = \
	"#version 430 core\n"
	"layout (local_size_x = 1) in;"
	"layout(std430, binding=0) writeonly buffer Pixels {vec4 fragColor[];};"
	"void main()"
	"{"
		"fragColor[gl_GlobalInvocationID.x] = vec4(1.0, 2.0, 3.0, 4.0);"
	"}" ;
	
int main()
{
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	glInit();
	int P = glCreateProgram();
	int CS = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(CS, 1, &ComputeShader, 0);
	glCompileShader(CS);
	glAttachShader(P, CS);
	glLinkProgram(P); 
	glUseProgram(P);
	unsigned int SSBO = 0;
	glGenBuffers(1, &SSBO);
	glBindBufferBase (GL_SHADER_STORAGE_BUFFER, 0, SSBO);
	glBufferData (GL_SHADER_STORAGE_BUFFER, 4 * sizeof(float), 0, GL_DYNAMIC_DRAW);
	glDispatchCompute(1, 1, 1);
	glMemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
	float *buffer = (float *) glMapBuffer (GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	glUnmapBuffer (GL_SHADER_STORAGE_BUFFER);
	printf("%f\n%f\n%f\n%f\n", buffer[0], buffer[1], buffer[2], buffer[3]);
	return 0;
}