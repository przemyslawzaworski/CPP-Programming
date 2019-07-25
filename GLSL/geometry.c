// cl geometry.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>

#define width 1280
#define height 720

typedef GLuint(WINAPI *PFNGLCREATEPROGRAMPROC) ();
typedef GLuint(WINAPI *PFNGLCREATESHADERPROC) (GLenum t);
typedef void(WINAPI *PFNGLSHADERSOURCEPROC) (GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(WINAPI *PFNGLCOMPILESHADERPROC) (GLuint s);
typedef void(WINAPI *PFNGLATTACHSHADERPROC) (GLuint p, GLuint s);
typedef void(WINAPI *PFNGLLINKPROGRAMPROC) (GLuint p);
typedef void(WINAPI *PFNGLUSEPROGRAMPROC) (GLuint p);
typedef int(WINAPI *PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef void(WINAPI *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void(WINAPI *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);

static const char* VertexShader = \
	"#version 450 \n"	
	"const vec3 vertices[4] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,1,0)};"
	"void main()"
	"{"	
		"gl_Position = vec4(vertices[gl_VertexID], 2);"
	"}";

static const char* GeometryShader = \
	"#version 450 \n"
	"#define num 8 \n"
	"layout(points) in;"
	"layout(line_strip, max_vertices = num + 1) out;"
	"const vec3 colors[4] = {vec3(1,0,0), vec3(0,1,0), vec3(0,0,1), vec3(1,1,1)};"
	"out vec3 hue;"
	"void main()"
	"{"
		"hue = colors[gl_PrimitiveIDIn];"
		"for (int i = 0; i <= num; i++)"
		"{"
			"float angle = 3.14159265 * 2.0 / num * i;"
			"vec4 offset = vec4(cos(angle)*0.3 , -sin(angle)*0.5 , 0.0, 0.0);"
			"gl_Position = gl_in[0].gl_Position + offset;"
			"EmitVertex();"
		"}"		
		"EndPrimitive();"
	"}";

static const char* FragmentShader = \
	"#version 450 \n"
	"out vec4 fragColor;"
	"in vec3 hue;"
	"void main()"
	"{"	
		"fragColor = vec4(hue, 1.0);"
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

int LoadShaders(const char* VS, const char* GS, const char* FS)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int sv = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B31);
	int sg = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8DD9);	
	int sf = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sv,1,&VS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sg,1,&GS,0);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sf,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sv);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sg);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sf);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sv);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sg);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sf);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(sv);
	Debug(sg);
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Geometry Shader"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Geometry Shader", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));	
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int SH = LoadShaders(VertexShader,GeometryShader, FragmentShader);	
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(SH);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glDrawArrays(GL_POINTS, 0, 4);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}