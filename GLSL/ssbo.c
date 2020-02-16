// cl ssbo.c opengl32.lib user32.lib gdi32.lib
// Shader Storage Buffer Object example
#include <windows.h>
#include <GL/gl.h>

#define width 1920
#define height 1080

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
typedef int (__stdcall *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (__stdcall *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);
typedef void (__stdcall *PFNGLUNIFORM2IPROC) (GLint s, GLint v0, GLint v1);

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
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORM1FPROC glUniform1f;
PFNGLUNIFORM2IPROC glUniform2i;

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
	glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
	glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");
	glUniform2i = (PFNGLUNIFORM2IPROC)wglGetProcAddress("glUniform2i");
}

static const char* ComputeShader = \
	"#version 430 core\n"
	"layout (local_size_x = 8, local_size_y = 8) in;"
	"layout(std430, binding=0) writeonly buffer Pixels {float image[];};"
	"uniform float iTime;"
	"uniform ivec2 iResolution;"
	
	"float encode(vec4 c)"
	"{"
		"int rgba = (int(c.w * 255.0) << 24) + (int(c.z * 255.0) << 16) + (int(c.y * 255.0) << 8) + int(c.x * 255.0);"
		"return intBitsToFloat(rgba);"
	"}"

	"float m(vec3 p)"
	"{"
		"float i = 0., s = 1., k = 0.;"
		"for(p.y += iTime * 0.3 ; i++<7.; s *= k ) p *= k = 1.5 / dot(p = mod(p - 1., 2.) - 1., p);"
		"return length(p)/s - .01;"
	"}"	
	
	"void main()"
	"{"
		"vec2 fragCoord = gl_GlobalInvocationID.xy;"
		"vec3 d = vec3((2.0*fragCoord-iResolution.xy)/iResolution.y,1)/6.0; "
		"vec3 o = vec3(1.0);"
		"vec4 c = vec4(0.0);"
		"while(c.w++<100.) o+= m(o)*d; "      
		"c.rgb += m(o - d) / pow(o.z,1.5) * 2.5;"
		"vec4 fragColor = vec4(c.rgb, 1.0);"
		"image[iResolution.x * gl_GlobalInvocationID.y + gl_GlobalInvocationID.x] = encode(clamp(vec4(fragColor.zyx,1.0),vec4(0.0),vec4(1.0)));"
	"}" ;
	
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
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_POPUP, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	ShowCursor(0);
	glInit();
	BITMAPINFO bmi = {{sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0}};
	unsigned char *buffer = (unsigned char *) malloc(width*height*4);
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
	glBufferData (GL_SHADER_STORAGE_BUFFER, width*height*4, 0, GL_DYNAMIC_DRAW);
	int iTime = glGetUniformLocation(P,"iTime");
	int iResolution = glGetUniformLocation(P,"iResolution");
	glUniform2i(iResolution, width, height);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glUniform1f(iTime, GetTickCount()*0.001f);
		glDispatchCompute(width/8, height/8, 1);
		glMemoryBarrier (GL_SHADER_STORAGE_BARRIER_BIT);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
		buffer = (unsigned char *) glMapBuffer (GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		StretchDIBits(hdc,0,0,width,height,0,0,width,height,buffer,&bmi,DIB_RGB_COLORS,SRCCOPY);
		glUnmapBuffer (GL_SHADER_STORAGE_BUFFER);	
	}
	free(buffer);
	return 0;
}