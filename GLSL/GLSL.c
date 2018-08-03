// gcc -x c -s -o GLSL.exe GLSL.c -mwindows -lopengl32 
#include <windows.h>
#include <GL/gl.h>

typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
 
static const char* FragmentShader = \
	"#version 450 \n"
	"layout (location=0) out vec4 color;"
	"void main()"
	"{"
		"color = vec4(gl_FragCoord.xy/vec2(1920,1080),0,1);"
	"}";

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s, 1, &FragmentShader, 0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p, s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(p);
	do
	{
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}