#include <windows.h>
#include "gl.h"

typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char *const *n, const GLint *i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
typedef GLint(APIENTRY* PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY* PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
 
static const char* FragmentShader = \
	"#version 450 \n"
	"layout (location=0) out vec4 color;"
	"uniform float time;"
	"float k = 9.0;"
	"float h(vec2 n)"
	"{" 
		"return fract(sin(dot(mod(n,k),vec2(16.7123,9.1414)))*43758.5453);"
	"}"
	"float n(vec2 p)"
	"{"
		"vec2 i = floor(p), u = fract(p);"
		"u = u*u*(3.-2.*u);"
		"return mix(mix(h(i),h(i+vec2(1,0)),u.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),u.x),u.y);"
	"}"
	"float t(vec2 u) "
	"{" 
		"float r = length( u ); "  
		"u = vec2( 1.0/r+(time*2.0),atan( u.y, u.x )/3.1415927 );  "
		"return n(k*u)*r;"
	"}"
	"void main()"
	"{"
		"vec2 u = (2.*gl_FragCoord.xy - vec2(1920,1080))/1080;"
		"color = vec4(0,0,t(u)*t(u),1);"
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
	GLint location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(p,"time");
	float S = GetTickCount(); 	
	do
	{
		float t = (GetTickCount()-S)*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);		
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
