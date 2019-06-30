// cl vs_quad.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>

#define width 1280.0f
#define height 720.0f

typedef GLuint(APIENTRY *PFNGLCREATEPROGRAMPROC) ();
typedef GLuint(APIENTRY *PFNGLCREATESHADERPROC) (GLenum t);
typedef void(APIENTRY *PFNGLSHADERSOURCEPROC) (GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(APIENTRY *PFNGLCOMPILESHADERPROC) (GLuint s);
typedef void(APIENTRY *PFNGLATTACHSHADERPROC) (GLuint p, GLuint s);
typedef void(APIENTRY *PFNGLLINKPROGRAMPROC) (GLuint p);
typedef void(APIENTRY *PFNGLUSEPROGRAMPROC) (GLuint p);
typedef int(APIENTRY *PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef void(APIENTRY *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void(APIENTRY *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);

static const char* VertexShader = \
	"#version 450 \n"	
	"out vec2 UV;"
	"const vec3 vertices[6] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(1,1,0), vec3(-1,1,0)};"
	"const vec2 uv[6] = {vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,0), vec2(1,1), vec2(0,1)};"
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"UV = uv[id];"
		"gl_Position = vec4(vertices[id], 2);"		
	"}";
	
static const char* FragmentShader = \
	"#version 450 \n"
	"out vec4 color;"
	"in vec2 UV;"
	"void main()"
	"{"	
		"color = vec4(UV,0.0,1.0);"
	"}";
	
void Debug(int sh)
{
	GLint isCompiled = 0;
	((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,0x8B82,&isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint length = 0;
		((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,0x8B84,&length);
		GLsizei q = 0;
		char* log = (char*)malloc(sizeof(char)*length);
		((PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog"))(sh,length,&q,log);
		if (length>1)
		{
			FILE *file = fopen ("debug.log","a");
			fprintf (file,"%s\n%s\n",(char*)glGetString(0x8B8C),log);
			fclose (file);
			ExitProcess(0);
		}
	}
}

int MakeShader(const char* VS, const char* FS)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int sv = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B31);	
	int sf = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sv,1,&VS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sf,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sv);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sf);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sv);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sf);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(sv);
	Debug(sf);
	return p;
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
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));	
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int PS = MakeShader(VertexShader,FragmentShader);	
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(PS);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}