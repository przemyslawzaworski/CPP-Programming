// cl quad.c opengl32.lib user32.lib gdi32.lib
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
typedef void(APIENTRY *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void(APIENTRY *PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void(APIENTRY *PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef void(APIENTRY *PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void(APIENTRY *PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void(APIENTRY *PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void(APIENTRY *PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef int(APIENTRY *PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef int(APIENTRY *PFNGLGETATTRIBLOCATIONPROC) (GLuint p, const char *n);
typedef void(APIENTRY *PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);
typedef void(APIENTRY *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void(APIENTRY *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);

static const GLfloat vertices[] = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f};
static const GLfloat uv[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
unsigned int VertexBuffer, UVBuffer, VertexArrayID;

static const char* VertexShader = \
	"#version 450 \n"
	"layout (location=0) in vec3 vertex;"
	"layout (location=1) in vec2 uv;"	
	"out vec2 UV;"	
	"void main()"
	"{"	
		"gl_Position = vec4(vertex, 2.0);"
		"UV = uv;"		
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
	int s1 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B31);	
	int s2 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s1,1,&VS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s2,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s1);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s2);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s1);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s2);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(s2);
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
	((PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays")) (1, &VertexArrayID);		
	((PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray")) (VertexArrayID);	
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &VertexBuffer);
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(vertices), vertices, 0x88E4);	
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &UVBuffer);	
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, UVBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(uv), uv, 0x88E4);	
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
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(0);
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(1);	
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, UVBuffer);		
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(1,2, GL_FLOAT, GL_FALSE, 0,(void*)0 );
		glDrawArrays(GL_TRIANGLES, 0, 2*3);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(0);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(1);		
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}