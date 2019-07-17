// cl stream.c opengl32.lib user32.lib gdi32.lib
// reference: https://www.vertexshaderart.com/art/ZSksx2deRsDocFDKT
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>

#define width 1280
#define height 720

typedef GLuint (WINAPI *PFNGLCREATEPROGRAMPROC) ();
typedef GLuint (WINAPI *PFNGLCREATESHADERPROC) (GLenum t);
typedef void (WINAPI *PFNGLSHADERSOURCEPROC) (GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void (WINAPI *PFNGLCOMPILESHADERPROC) (GLuint s);
typedef void (WINAPI *PFNGLATTACHSHADERPROC) (GLuint p, GLuint s);
typedef void (WINAPI *PFNGLLINKPROGRAMPROC) (GLuint p);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC) (GLuint p);
typedef int (WINAPI *PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef void (WINAPI *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void (WINAPI *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (WINAPI *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);

static const char* VertexShader = \
	"#version 450 \n"
	"out vec3 v_color; \n"
	"uniform float time;"

	"vec3 rgb(vec3 c)"
	"{"
		"c = vec3(c.x, clamp(c.yz, 0.0, 1.0));"
		"vec4 K = vec4(1.0, 0.666, 0.333, 3.0);"
		"vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);"
		"return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);"
	"}"
	
	"void main()"
	"{"	
		"vec2 resolution = vec2(1280, 720);"
		"int vertexId = gl_VertexID;"
		"float point = mod(floor(vertexId / 2.0) + mod(float(vertexId), 2.0) * 5.0, 4.0);"
		"float count = floor(vertexId / 8.0);"
		"float offset = count * 0.02;"
		"float angle = point * radians(180.) * 2.0 / 4.0 + offset;"
		"float radius = 0.2 * pow(1.0, .0);"
		"float c = cos(angle + time) * radius;"
		"float s = sin(angle + time) * radius;"
		"float orbitAngle =  count * 0.0;"
		"float innerRadius = count * 0.001;"
		"float oC = sin(orbitAngle + time * 0.3 + count * 0.1) * innerRadius;"
		"float oS = tan(orbitAngle + time + count * 0.1) * innerRadius;"
		"vec2 aspect = vec2(1, resolution.x / resolution.y);"
		"vec2 xy = vec2(oC + c,oS + s);"
		"gl_Position = vec4(xy * aspect, 0, 1);"
		"float hue = (time * 0.01 + count * 1.001);"
		"v_color = vec3(rgb(vec3(hue, 1, 1)));"
	"}";
	
static const char* FragmentShader = \
	"#version 450 \n"
	"in vec3 v_color;"
	"out vec4 color;"
	"void main()"
	"{"	
		"color = vec4(v_color,1.0);"
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

int MakeShaders(const char* VS, const char* FS)
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
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	SetPixelFormat(hdc, ChoosePixelFormat(hdc,&pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));	
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int SH = MakeShaders(VertexShader, FragmentShader);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(SH);
	int time = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(SH, "time");
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glEnable(0x8642);
		((PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f"))(time, GetTickCount()*0.001f);
		glDrawArrays(GL_LINES, 0, 9623);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}