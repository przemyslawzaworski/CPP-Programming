// Compile from Visual Studio command line: cl nvstream.c opengl32.lib user32.lib gdi32.lib
// Require NVIDIA RTX GPU.
// Reference: https://www.vertexshaderart.com/art/ZSksx2deRsDocFDKT
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>

typedef GLuint (WINAPI *PFNGLCREATEPROGRAMPROC) ();
typedef GLuint (WINAPI *PFNGLCREATESHADERPROC) (GLenum t);
typedef void (WINAPI *PFNGLSHADERSOURCEPROC) (GLuint s, GLsizei c, const char*const*str, const GLint* i);
typedef void (WINAPI *PFNGLCOMPILESHADERPROC) (GLuint s);
typedef void (WINAPI *PFNGLATTACHSHADERPROC) (GLuint p, GLuint s);
typedef void (WINAPI *PFNGLLINKPROGRAMPROC) (GLuint p);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC) (GLuint p);
typedef void (WINAPI *PFNGLGETSHADERIVPROC) (GLuint s, GLenum v, GLint *p);
typedef void (WINAPI *PFNGLGETSHADERINFOLOGPROC) (GLuint s, GLsizei b, GLsizei *l, char *i);
typedef void (WINAPI *PFNGLDRAWMESHTASKSNVPROC) (GLuint f, GLuint c);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (WINAPI *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);
typedef int (WINAPI *PFNWGLSWAPINTERVALEXTPROC) (int i);

#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_SHADING_LANGUAGE_VERSION       0x8B8C
#define GL_TASK_SHADER_NV                 0x955A
#define GL_MESH_SHADER_NV                 0x9559
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_PROGRAM_POINT_SIZE             0x8642

PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNGLDRAWMESHTASKSNVPROC glDrawMeshTasksNV;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORM1FPROC glUniform1f;
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;

void glInit()
{
	glCreateProgram = (PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram");
	glCreateShader = (PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader");
	glShaderSource = (PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource");
	glCompileShader = (PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader");
	glAttachShader = (PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader");
	glLinkProgram = (PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram");
	glUseProgram = (PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram");
	glGetShaderiv = (PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv");
	glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog");
	glDrawMeshTasksNV = (PFNGLDRAWMESHTASKSNVPROC)wglGetProcAddress("glDrawMeshTasksNV");
	glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
	glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");	
	wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");	
}

static const char* TaskShader = \
	"#version 460 \n"
	"#extension GL_NV_mesh_shader : enable \n"
	"layout(local_size_x = 32) in;"
	"void main()"
	"{"
		"gl_TaskCountNV = 256;"
	"}";
	
static const char* MeshShader = \
	"#version 460 \n"
	"#extension GL_NV_mesh_shader : enable \n"
	"layout(local_size_x = 32) in;"
	"layout(max_vertices = 32) out;"
	"layout(max_primitives = 16) out;"
	"layout(lines) out;"
	"layout(location = 0) out Interpolants{vec3 v_color;} OUT[];"
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
		"uint laneID = gl_LocalInvocationID.x;"
		"uint baseID = gl_GlobalInvocationID.x;"
		"vec2 resolution = vec2(1280, 720);"
		"int vertexId = int(baseID);"
		"float point = mod(floor(vertexId / 2.0) + mod(float(vertexId), 2.0) * 5.0, 4.0);"
		"float count = floor(vertexId / 8.0);"
		"float offset = count * 0.02;"
		"float angle = point * radians(180.) * 2.0 / 4.0 + offset;"
		"float radius = 0.2;"
		"float c = cos(angle + time) * radius;"
		"float s = sin(angle + time) * radius;"
		"float innerRadius = count * 0.001;"
		"float oC = sin(time * 0.3 + count * 0.1) * innerRadius;"
		"float oS = tan(time + count * 0.1) * innerRadius;"
		"vec2 aspect = vec2(1, resolution.x / resolution.y);"
		"vec2 xy = vec2(oC + c,oS + s);"		
		"gl_MeshVerticesNV[laneID].gl_PointSize = 1;"
		"gl_MeshVerticesNV[laneID].gl_Position = vec4(xy * aspect, 0, 1);"
		"gl_PrimitiveIndicesNV[laneID] = laneID;"
		"gl_PrimitiveCountNV = 16;"
		"float hue = (time * 0.01 + count * 1.001);"
		"OUT[laneID].v_color = vec3(rgb(vec3(hue, 1, 1)));"
	"}";
	
static const char* FragmentShader = \
	"#version 460 \n"
	"#extension GL_NV_fragment_shader_barycentric : enable\n"
	"layout(location = 0) in Interpolants{vec3 v_color;} IN;"
	"out vec4 color;"
	"void main()"
	"{"
		"color = vec4(IN.v_color, 1.0);"
	"}";

void Debug(int sh)
{
	GLint isCompiled = 0;
	glGetShaderiv(sh, GL_LINK_STATUS, &isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint length = 0;
		glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &length);
		GLsizei q = 0;
		char* log = (char*)malloc(sizeof(char)*length);
		glGetShaderInfoLog(sh,length,&q,log);
		if (length>1)
		{
			FILE *file = fopen ("debug.log","a");
			fprintf (file,"%s\n%s\n",(char*)glGetString(GL_SHADING_LANGUAGE_VERSION),log);
			fclose (file);
			ExitProcess(0);
		}
	}
}

int MakeShaders(const char* TS, const char* MS, const char* FS)
{
	int p = glCreateProgram();
	int st = glCreateShader(GL_TASK_SHADER_NV);
	int sm = glCreateShader(GL_MESH_SHADER_NV);
	int sf = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(st, 1, &TS, 0);
	glShaderSource(sm, 1, &MS, 0);
	glShaderSource(sf, 1, &FS, 0);
	glCompileShader(st);
	glCompileShader(sm);
	glCompileShader(sf);
	glAttachShader(p, st);
	glAttachShader(p, sm);
	glAttachShader(p, sf);
	glLinkProgram(p);
	Debug(st);
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "RTX Task and Mesh Shader Demo"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "RTX Task and Mesh Shader Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glInit();
	wglSwapIntervalEXT(0);
	int SH = MakeShaders(TaskShader, MeshShader, FragmentShader);
	glUseProgram(SH);
	int time = glGetUniformLocation(SH, "time");
	glEnable(GL_PROGRAM_POINT_SIZE);	
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glUniform1f(time, GetTickCount()*0.001f);
		glDrawMeshTasksNV(0, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}