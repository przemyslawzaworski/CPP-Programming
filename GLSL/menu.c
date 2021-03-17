// gcc -s -o menu.exe menu.c -mwindows -lopengl32
#include <windows.h>
#include <GL/gl.h>

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

float ScreenWidth = 1920.0f;
float ScreenHeight = 1080.0f;
HWND Button, ComboBox;

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
	
	"const mat3 rotationMatrix = mat3(1.0,0.0,0.0,0.0,0.7,-0.7,0.0,0.7,0.7);"

	"float hash(vec2 p)"
	"{"
		"vec3 p3  = fract(vec3(p.xyx) * 10.21);"
		"p3 += dot(p3, p3.yzx + 19.19);"
		"return fract((p3.x + p3.y) * p3.z);"
	"}"
	"float noise (vec2 P)"
	"{"
		"float size = 256.0;"
		"float s = (1.0 / size);"
		"vec2 pixel = P * size + 0.5;"   
		"vec2 f = fract(pixel);"
		"pixel = (floor(pixel) / size) - vec2(s/2.0, s/2.0);"
		"float C11 = hash(pixel + vec2( 0.0, 0.0));"
		"float C21 = hash(pixel + vec2( s, 0.0));"
		"float C12 = hash(pixel + vec2( 0.0, s));"
		"float C22 = hash(pixel + vec2( s, s));"
		"float x1 = mix(C11, C21, f.x);"
		"float x2 = mix(C12, C22, f.x);"
		"return mix(x1, x2, f.y);"
	"}"
	"float fbm( vec2 p )"
	"{"
		"float a = 0.5, b = 0.0, t = 0.0;"
		"for (int i=0; i<5; i++)"
		"{"
			"b *= a; t *= a;"
			"b += noise(p);"
			"t += 1.0; p /= 2.0;"
		"}"
		"return b /= t;"
	"}"
	"float map( vec3 p )"
	"{ "
		"float h = p.y - 20.0 * fbm(p.xz*0.003);"
		"return max( min( h, 0.55), p.y-20.0 );"
	"}"
	"bool raymarch( inout vec3 ro, vec3 rd)"
	"{"
		"float t = 0.0;"
		"for (int i=0; i<128; i++)"
		"{"
			"float d = map(ro+rd*t);"
			"t+=d;"
			"if (d<t*0.001)"
			"{"
				"ro+=t*rd;"
				"return true;"
			"}"
		"}"
		"return false;"
	"}"
	"vec3 shading( vec3 ro, vec3 rd )"
	"{"
		"vec3 c = vec3(rd.y*2.0) * 0.1;"
		"vec3 sk = ro;"
		"if (raymarch(ro,rd))"
		"{"
			"vec2 p = ro.xz;"
			"vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p);"
			"float s = min(g.x, g.y);"
			"float f = min(length(ro-sk)/64.,1.);"
			"return mix(2.0-vec3(s,s,s), c, f);"
		"}"
		"return vec3(0.0);"
	"}"	
	"void main()"
	"{"
		"vec2 fragCoord = gl_FragCoord.xy;"
		"vec2 uv = (2.*fragCoord.xy - iResolution.xy)/iResolution.y;"
		"vec3 ro = vec3(0.5,25.+sin(iTime)*5.0,iTime * 5.0);"
		"vec3 rd = normalize(vec3(uv,2.0)) * rotationMatrix;"  
		"fragColor = vec4(shading(ro,rd), 1.0);"
	"}";


static LRESULT CALLBACK WindowProcRenderSetup(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_CLOSE)
	{
		PostQuitMessage(0); ExitProcess(0); return 0;
	}
	if (uMsg == WM_COMMAND)
	{
		if( (HWND) lParam == Button )
		{
			DWORD length = GetWindowTextLength( ComboBox );
			LPSTR buffer =( LPSTR ) GlobalAlloc( GPTR, length + 1 );
			GetWindowText( ComboBox, buffer, length + 1 );
			char *array[2];
			char *temp = strtok (buffer, "x");
			array[0] = temp;
			temp = strtok (NULL, "x");
			array[1] = temp;	
			ScreenWidth = atoi(array[0]);
			ScreenHeight = atoi(array[1]);
			PostQuitMessage( 0 );
			return 0;
		}
		else
		{
			return DefWindowProc(hWnd, uMsg, wParam, lParam);
		}
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}	
}

void RenderSetup (HINSTANCE hInstance)
{
	int exit = 0;
	MSG msg;
	WNDCLASS winMenu = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProcRenderSetup, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Setup"};
	RegisterClass(&winMenu);
	HWND menu = CreateWindowEx(0, winMenu.lpszClassName, "Setup", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 320, 240, 320, 240, 0, 0, 0, 0);
	Button = CreateWindowEx( 0, "BUTTON", "OK", WS_CHILD | WS_VISIBLE, 50, 100, 150, 30, menu, NULL, hInstance, NULL );
	ComboBox = CreateWindowEx( WS_EX_CLIENTEDGE, "COMBOBOX", NULL, WS_CHILD | WS_VISIBLE | WS_BORDER |CBS_DROPDOWNLIST, 50, 50, 150, 200, menu, 0, hInstance, 0 );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "800x600" );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1280x720" );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1280x1024" );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1366x768" );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1600x900" );	
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1680x1050" );	
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "1920x1080" );
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "2560x1440" );	
	SendMessage( ComboBox, CB_ADDSTRING, 0,( LPARAM ) "3840x2160" );	
	SendMessage( ComboBox, CB_SETCURSEL, 0, 0);
	HWND text = CreateWindowEx( 0, "STATIC", NULL, WS_CHILD | WS_VISIBLE | SS_LEFT, 50, 25, 150, 20, menu, NULL, hInstance, NULL );
	SetWindowText( text, "Resolution: " );
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
	}
	DestroyWindow(text);
	DestroyWindow(Button);
	DestroyWindow(ComboBox);
	DestroyWindow(menu);	
}

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
	RenderSetup (hInstance);
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