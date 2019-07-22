// cl bindless.c opengl32.lib user32.lib gdi32.lib
// Code written by: Przemyslaw Zaworski
// Program renders 1024 textures (heightmaps with 256x256 resolution) with only one draw call.
// Download textures:
// https://mega.nz/#!Msw1ESbI!8CSRZFj8TFlQ8V5nv7iOcomxlKpSyz6i3RxFBw1dpLI
// More info:
// http://developer.download.nvidia.com/opengl/specs/GL_NV_bindless_texture.txt
// https://www.geeks3d.com/20120511/nvidia-gtx-680-opengl-bindless-textures-demo/
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>

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
typedef uint64_t (WINAPI *PFNGLGETTEXTUREHANDLENVPROC) (GLuint texture);
typedef void (WINAPI *PFNGLMAKETEXTUREHANDLERESIDENTNVPROC) (uint64_t handle);
typedef void (WINAPI *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (WINAPI *PFNGLBINDBUFFERPROC) (GLenum t, GLuint buffer);
typedef void (WINAPI *PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t size, const void *data, GLenum u);
typedef void (WINAPI *PFNGLBINDBUFFERRANGEPROC) (GLenum t, GLuint i, GLuint b, int o, ptrdiff_t size);
typedef void (WINAPI *PFNGLBUFFERSUBDATAPROC) (GLenum t, int o, ptrdiff_t size, const void *data);

#define GL_UNIFORM_BUFFER                 0x8A11
#define GL_STATIC_DRAW                    0x88E4
#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_SHADING_LANGUAGE_VERSION       0x8B8C
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31

PFNGLCREATEPROGRAMPROC glCreateProgram;
PFNGLCREATESHADERPROC glCreateShader;
PFNGLSHADERSOURCEPROC glShaderSource;
PFNGLCOMPILESHADERPROC glCompileShader;
PFNGLATTACHSHADERPROC glAttachShader;
PFNGLLINKPROGRAMPROC glLinkProgram;
PFNGLUSEPROGRAMPROC glUseProgram;
PFNGLGETSHADERIVPROC glGetShaderiv;
PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog;
PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT;
PFNGLGETTEXTUREHANDLENVPROC glGetTextureHandleNV;
PFNGLMAKETEXTUREHANDLERESIDENTNVPROC glMakeTextureHandleResidentNV;
PFNGLGENBUFFERSPROC glGenBuffers;
PFNGLBINDBUFFERPROC glBindBuffer;
PFNGLBUFFERDATAPROC glBufferData;
PFNGLBINDBUFFERRANGEPROC glBindBufferRange;
PFNGLBUFFERSUBDATAPROC glBufferSubData;

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
	wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
	glGetTextureHandleNV = (PFNGLGETTEXTUREHANDLENVPROC)wglGetProcAddress("glGetTextureHandleNV");
	glMakeTextureHandleResidentNV = (PFNGLMAKETEXTUREHANDLERESIDENTNVPROC)wglGetProcAddress("glMakeTextureHandleResidentNV");
	glGenBuffers = (PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"); 
	glBindBuffer = (PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer");
	glBufferData = (PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData");
	glBindBufferRange = (PFNGLBINDBUFFERRANGEPROC)wglGetProcAddress("glBindBufferRange");
	glBufferSubData = (PFNGLBUFFERSUBDATAPROC)wglGetProcAddress("glBufferSubData");	
}

static const char* VertexShader = \
	"#version 450 \n"
	"#extension GL_NV_bindless_texture : require \n"
	"#extension GL_NV_gpu_shader5 : require \n"	
	"out vec2 UV;"
	"const vec3 vertices[6] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(1,1,0), vec3(-1,1,0)};"
	"const vec2 uv[6] = {vec2(0,0), vec2(1,0), vec2(0,1), vec2(1,0), vec2(1,1), vec2(0,1)};"
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"UV = uv[id];"
		"gl_Position = vec4(vertices[id], 1);"
	"}";
	
static const char* FragmentShader = \
	"#version 450 \n"
	"#extension GL_NV_bindless_texture : require \n"
	"#extension GL_NV_gpu_shader5 : require \n"
	"out vec4 fragColor;"
	"in vec2 UV;"
	"uniform SamplersNV { uint64_t allTheSamplers[1024]; };"
	"vec3 grid(vec2 p)"
	"{"
		"p = p * 32.0;"
		"vec2 g = abs(fract(p - 0.5) - 0.5) / fwidth(p);"
		"float s = min(g.x, g.y);"
		"return vec3(s);"
	"}"
	"void main ()"
	"{"
		"vec2 uv = UV * vec2(32, 32);"
		"int x = int(uv.x);"
		"int y = int(uv.y);"
		"int index = y * 32 + x;"
		"sampler2D s = sampler2D(allTheSamplers[index]);"
		"fragColor = min(vec4(texture(s, uv).rgb, 1.0) ,vec4(grid(UV)*0.3, 1.0));"
	"}";

char* Concat( char* s1, char* s2 )
{
	char* s3 = (char*) malloc(1+strlen(s1)+strlen(s2));
	strcpy(s3, s1);
	strcat(s3, s2);
	return s3;
}

void LoadTextures(char *path)
{
	unsigned int textures[1024];
	uint64_t texHandles[1024];
	glGenTextures(1024, textures);
	for (int i=0; i<1024; i++)
	{
		int width, height;
		char buffer[128]; 
		char counter[5];
		itoa(i, counter, 10);
		FILE *file = fopen(Concat(path, Concat(counter,".ppm")), "rb");
		fgets(buffer, sizeof(buffer), file);
		do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
		sscanf (buffer, "%d %d", &width, &height);
		do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
		int size = width * height * 4 * sizeof(GLubyte);
		GLubyte *Texels  = (GLubyte *)malloc(size);
		for (int i = 0; i < size; i++) 
		{
			Texels[i] = ((i % 4) < 3 ) ? (GLubyte) fgetc(file) : (GLubyte) 255;
		}
		fclose(file);
		glBindTexture(GL_TEXTURE_2D, textures[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, Texels);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		texHandles[i] = glGetTextureHandleNV(textures[i]);
		glMakeTextureHandleResidentNV(texHandles[i]);
	}
	unsigned int container;
	glGenBuffers(1, &container); 
	glBindBuffer(GL_UNIFORM_BUFFER, container);
	glBufferData(GL_UNIFORM_BUFFER, 1024 * sizeof(uint64_t), NULL, GL_STATIC_DRAW);
	glBindBufferRange(GL_UNIFORM_BUFFER, 0, container, 0, 1024 * sizeof(uint64_t));
	glBufferSubData(GL_UNIFORM_BUFFER, 0, 1024 * sizeof(uint64_t), texHandles);
}

void DebugShader(int sh)
{
	GLint isCompiled = 0;
	glGetShaderiv(sh,GL_LINK_STATUS,&isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint length = 0;
		glGetShaderiv(sh,GL_INFO_LOG_LENGTH,&length);
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

int LoadShaders(const char* VS, const char* FS)
{
	int p = glCreateProgram();
	int sv = glCreateShader(GL_VERTEX_SHADER);
	int sf = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(sv, 1, &VS, 0);
	glShaderSource(sf, 1, &FS, 0);	
	glCompileShader(sv);
	glCompileShader(sf);
	glAttachShader(p,sv);
	glAttachShader(p,sf);
	glLinkProgram(p);
	DebugShader(sv);
	DebugShader(sf);
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
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "Loading...", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, 1920, 1080, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glInit();	
	wglSwapIntervalEXT(0);
	int SH = LoadShaders(VertexShader,FragmentShader);	
	glUseProgram(SH);
	LoadTextures("heightmaps\\");
	SetWindowTextA(hwnd, "NVIDIA Bindless Textures Demo");
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