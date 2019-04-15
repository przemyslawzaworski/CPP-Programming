// Compile with Visual Studio: cl redblue.c opengl32.lib user32.lib gdi32.lib
// GLSL to Asm: cgc -oglsl -profile fp30 test.glsl 
// ARB Assembly Language details: http://www.renderguild.com/gpuguide.pdf

#include <windows.h>
#include <GL/gl.h>
#include <string.h>

typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef void(APIENTRY* PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const void *string);
 
static const char* FS = \
	"!!ARBfp1.0\n"
	"TEMP R0, H0;"
	"MUL R0.x, fragment.position, {0.00052};"
	"SGE H0.x, {0.5}, R0;"
	"MAD result.color, H0.x, {1,0,-1,0}, {0,0,1,1};"
	"END";

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	glEnable(0x8804);
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);	
	((PFNGLPROGRAMSTRINGARBPROC)wglGetProcAddress("glProgramStringARB")) (0x8804, 0x8875, strlen(FS), FS);
	do
	{
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
