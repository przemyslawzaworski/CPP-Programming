// g++ -s -o GLSL.exe GLSL.cpp -IGL/inc -LGL/lib -mwindows -lopengl32 -lglew32
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <glew.h>

const static char* FragmentShader = \
"#version 130\n"
"uniform float time;"
"void main()"
"{"
	"vec2 uv = gl_FragCoord.xy / vec2(1920,1080);"
	"gl_FragColor = vec4(uv,sin(time)*0.5+0.5,1.0);"
"}" ;

int main()
{ 
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = {0,0,PFD_DOUBLEBUFFER,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	DEVMODE setup = {{0},0,0,156,0,0x001c0000,{0},0,0,0,0,0,{0},0,32,1920,1080,{0},0,0,0};	
	HDC hdc = GetDC(CreateWindow("static",0,WS_POPUP|WS_VISIBLE|WS_MAXIMIZE,0,0,0,0,0,0,0,0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	const int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	const int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(GL_FRAGMENT_SHADER);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s,1,&FragmentShader,0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(p);
	GLint location = ((PFNGLGETUNIFORMLOCATIONARBPROC)wglGetProcAddress("glGetUniformLocation"))(p,"time");
	do
	{ 
		glRects(-1,-1,1,1);
		GLfloat t = GetTickCount()*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);
		SwapBuffers(hdc);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	ExitProcess(0);
	return 0;
}
