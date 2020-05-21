// Author: Przemyslaw Zaworski 18.07.2019
// Compile from Visual Studio command line: cl fireball.c opengl32.lib user32.lib gdi32.lib
// Demo features procedural point cloud with drawn raytraced sphere on it.
// Require NVIDIA RTX GPU.
// Total vertex count = gl_TaskCountNV * gl_PrimitiveCountNV = 1280 * 720 * 0.25 = 230400
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
	"layout(local_size_x = 8) in;"
	"void main()"
	"{"
		"gl_TaskCountNV = 28800;"
	"}";
	
static const char* MeshShader = \
	"#version 460 \n"
	"#extension GL_NV_mesh_shader : enable \n"
	"layout(local_size_x = 8) in;"
	"layout(max_vertices = 8) out;"
	"layout(max_primitives = 8) out;"
	"layout(points) out;"
	"layout(location = 0) out Interpolants{vec3 v_color;} OUT[];"
	"uniform float time;"
	
	"vec3 hash( vec2 p )"
	"{"
		"vec3 q = vec3( dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)), dot(p,vec2(419.2,371.9)) );"
		"return fract(sin(q)*43758.5453);"
	"}"

	"vec3 noise(vec2 p)"
	"{"
		"vec2 ip = floor(p);"
		"vec2 u = fract(p);"
		"u = u*u*(3.0-2.0*u);"
		"vec3 res = mix(mix(hash(ip),hash(ip+vec2(1,0)),u.x), mix(hash(ip+vec2(0,1)),hash(ip+vec2(1,1)),u.x),u.y);"
		"return res*res;"
	"}"

	"vec3 fbm(vec2 x)"
	"{"
		"vec3 v = vec3(0.0);"
		"vec3 a = vec3(0.5);"
		"mat2 rot = mat2(0.87, 0.48, -0.48, 0.87);"
		"for (int i = 0; i < 6; ++i) "
		"{"
			"v += a * noise(x);"
			"x = rot * x * 2.0 + vec2(100);"
			"a *= 0.5;"
		"}"
		"return v;"
	"}"

	"vec3 surface (vec2 uv)"
	"{"
		"uv.y = uv.y + time;"
		"return fbm(5.0 * uv);"
	"}"

	"float sphere(vec3 ro, vec3 rd)"
	"{"
		"float b = dot(ro,rd);"
		"float c = dot(ro,ro)-1.0;"
		"float h = b*b-c;"
		"return (h<0.0)?-1.0:-b-sqrt(h);"
	"}"

	"vec3 raycast(vec3 ro, vec3 rd, vec2 p)"
	"{"
		"float t = sphere(ro,rd);"
		"if (t > 0.0)"
		"{"
			"vec3 d = ro+rd*t;"
			"p = vec2(acos(d.y/length(d)), atan(d.z,d.x));"
			"return (surface(p));  "      
		"}"
		"return vec3(0.0);"
	"}"

	"vec3 radialblur (vec3 ro, vec2 uv, int factor)"
	"{"
		"vec3 col = vec3(0.0,0.0,0.0);"
		"vec2 d = (vec2(0.0,0.0)-uv)/float(factor);"
		"float w = 1.0;"
		"vec2 s = uv;"
		"for( int i=0; i<100; i++ )"
		"{"
			"vec3 res = raycast(ro,normalize(vec3(s,2.0)),s);"
			"col += w*smoothstep( 0.0, 1.0, res );"
			"w *= .98;"
			"s += d;"
	   " }"
		"return ( col * 6.0 / float(factor)); "   
	"}"
	
	"void main()"
	"{"
		"uint laneID = gl_LocalInvocationID.x;"
		"uint baseID = gl_GlobalInvocationID.x;"
		"vec2 resolution = vec2(1280, 720);"
		"int vertexId = int(baseID);"
		"int vertexCount = 230400;"
		"float grid = floor(sqrt(vertexCount));"
		"float u = mod(vertexId, grid) / grid * 2.0 - 1.0;"
		"float v = ceil(vertexId / grid) / grid * 2.0 - 1.0;"  
		"vec3 ro = vec3(0.0, 0.0, -7.0);" 
		"vec2 aspect = vec2(1, resolution.x / resolution.y);"		
		"gl_MeshVerticesNV[laneID].gl_PointSize = 3;"
		"gl_MeshVerticesNV[laneID].gl_Position = vec4(vec2(u,v) * aspect, 0.0, 1.0);"
		"gl_PrimitiveIndicesNV[laneID] = laneID;"
		"gl_PrimitiveCountNV = 8;"
		"OUT[laneID].v_color = radialblur(ro, vec2(u,v), 100);"
	"}";
	
static const char* FragmentShader = \
	"#version 460 \n"
	"layout(location = 0) in Interpolants{vec3 v_color;} IN;"
	"out vec4 fragColor;"
	"void main()"
	"{"
		"fragColor = vec4(IN.v_color, 1.0);"
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
	float S = GetTickCount()*0.001f;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glClear(GL_COLOR_BUFFER_BIT);
		glUniform1f(time, GetTickCount()*0.001f - S);
		glDrawMeshTasksNV(0, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}