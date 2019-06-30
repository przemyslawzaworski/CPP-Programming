// cl ms_triangle.c opengl32.lib user32.lib gdi32.lib
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
typedef void(WINAPI *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void(WINAPI *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);
typedef void(WINAPI *PFNGLDRAWMESHTASKSNVPROC) (GLuint f, GLuint c);

static const char* MeshShader = \
	"#version 450 \n"
	"#extension GL_NV_mesh_shader : enable\n"
	"layout(local_size_x = 3) in;"
	"layout(max_vertices = 64) out;"
	"layout(max_primitives = 126) out;"
	"layout(triangles) out;"
	"const vec3 vertices[3] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(0,1,0)};"
	"void main()"
	"{"
		"uint id = gl_LocalInvocationID.x;"
		"gl_MeshVerticesNV[id].gl_Position = vec4(vertices[id], 2);"
		"gl_PrimitiveIndicesNV[id] = id;"
		"gl_PrimitiveCountNV = 1;"
	"}";
	
static const char* FragmentShader = \
	"#version 450 \n"
	"#extension GL_NV_fragment_shader_barycentric : enable\n"
	"out vec4 color;"
	"void main()"
	"{"	
		"color = vec4(gl_BaryCoordNV, 1.0);"
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

int MakeShaders(const char* MS, const char* FS)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int sm = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x9559);	
	int sf = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sm,1,&MS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(sf,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sm);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(sf);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sm);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,sf);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(sm);
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Mesh Shader Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Mesh Shader Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	int PS = MakeShaders(MeshShader,FragmentShader);
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
		((PFNGLDRAWMESHTASKSNVPROC)wglGetProcAddress("glDrawMeshTasksNV"))(0,1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}