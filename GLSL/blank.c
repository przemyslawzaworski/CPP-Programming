// Compile with Visual Studio command line: cl blank.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>

#define ScreenWidth 1920.0f
#define ScreenHeight 1080.0f
#define GL_VERTEX_SHADER         0x8B31
#define GL_FRAGMENT_SHADER       0x8B30
#define GL_VERTEX_SHADER_BIT     0x00000001
#define GL_FRAGMENT_SHADER_BIT   0x00000002

typedef GLuint (__stdcall *PFNGLCREATESHADERPROGRAMVPROC) (GLenum type, GLsizei count, const char *const*strings);
typedef void (__stdcall *PFNGLGENPROGRAMPIPELINESPROC) (GLsizei n, GLuint *pipelines);
typedef void (__stdcall *PFNGLBINDPROGRAMPIPELINEPROC) (GLuint pipeline);
typedef void (__stdcall *PFNGLUSEPROGRAMSTAGESPROC) (GLuint pipeline, GLbitfield stages, GLuint program);
typedef void (__stdcall *PFNGLPROGRAMUNIFORM1FPROC) (GLuint program, GLint location, GLfloat v0);
typedef void (__stdcall *PFNGLPROGRAMUNIFORM2FPROC) (GLuint program, GLint location, GLfloat v0, GLfloat v1);

PFNGLCREATESHADERPROGRAMVPROC glCreateShaderProgramv;
PFNGLGENPROGRAMPIPELINESPROC glGenProgramPipelines;
PFNGLBINDPROGRAMPIPELINEPROC glBindProgramPipeline;
PFNGLUSEPROGRAMSTAGESPROC glUseProgramStages;
PFNGLPROGRAMUNIFORM1FPROC glProgramUniform1f;
PFNGLPROGRAMUNIFORM2FPROC glProgramUniform2f;

void glInit()
{
	glCreateShaderProgramv = (PFNGLCREATESHADERPROGRAMVPROC)wglGetProcAddress ("glCreateShaderProgramv");
	glGenProgramPipelines = (PFNGLGENPROGRAMPIPELINESPROC)wglGetProcAddress ("glGenProgramPipelines");
	glBindProgramPipeline = (PFNGLBINDPROGRAMPIPELINEPROC)wglGetProcAddress ("glBindProgramPipeline");
	glUseProgramStages = (PFNGLUSEPROGRAMSTAGESPROC)wglGetProcAddress ("glUseProgramStages");
	glProgramUniform1f = (PFNGLPROGRAMUNIFORM1FPROC)wglGetProcAddress ("glProgramUniform1f");
	glProgramUniform2f = (PFNGLPROGRAMUNIFORM2FPROC)wglGetProcAddress ("glProgramUniform2f");	
}

static const char* VertexShader = \
	"#version 450\n"
	"layout (location=0) in vec3 position;"
	"void main()"
	"{"
		"gl_Position = vec4(position, 1.0);"
	"}";

static const char* FragmentShader = \
	"#version 450\n"
	"layout (location=0) out vec4 fragColor;"
	"layout (location=1) uniform float iTime;"
	"layout (location=2) uniform vec2 iResolution;"	
	"void main()"
	"{"
		"vec2 fragCoord = gl_FragCoord.xy;"
		"vec2 uv = fragCoord/iResolution.xy;"
		"vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));"
		"fragColor = vec4(col,1.0);"
	"}";

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE))
	{
		PostQuitMessage(0); return 0;
	}
	else { return DefWindowProc(hWnd, uMsg, wParam, lParam); }
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_POPUP, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	PIXELFORMATDESCRIPTOR pfd = { 0, 0, PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	ShowCursor(0);
	glInit();
	int vs = glCreateShaderProgramv (GL_VERTEX_SHADER, 1, &VertexShader);
	int fs = glCreateShaderProgramv (GL_FRAGMENT_SHADER, 1, &FragmentShader);
	unsigned int p;
	glGenProgramPipelines (1, &p);
	glBindProgramPipeline (p);
	glUseProgramStages(p, GL_VERTEX_SHADER_BIT, vs);
	glUseProgramStages(p, GL_FRAGMENT_SHADER_BIT, fs);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float time = GetTickCount() * 0.001f;
		glProgramUniform1f (fs, 1, time);
		glProgramUniform2f (fs, 2, ScreenWidth, ScreenHeight);
		glRects (-1, -1, 1, 1);
		wglSwapLayerBuffers (hdc, WGL_SWAP_MAIN_PLANE); 
	}
	return 0;
}