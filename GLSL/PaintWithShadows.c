// gcc -s -o PaintWithShadows.exe PaintWithShadows.c -mwindows -lopengl32
// Written by: Przemyslaw Zaworski
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdbool.h>

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
typedef void (WINAPI *PFNGLGENFRAMEBUFFERSPROC) (GLsizei n, GLuint *f);
typedef void (WINAPI *PFNGLBINDFRAMEBUFFEREXTPROC) (GLenum t, GLuint f);
typedef void (WINAPI *PFNGLFRAMEBUFFERTEXTUREPROC) (GLenum t, GLenum a, GLuint s, GLint l);
typedef GLint(WINAPI *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (WINAPI *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);
typedef void (WINAPI *PFNGLUNIFORM1IPROC) (GLint l, GLint v);
typedef void (WINAPI *PFNGLUNIFORM4FPROC) (GLint l, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef void (WINAPI *PFNGLCLAMPCOLORPROC) (GLenum t, GLenum c);

#define GL_LINK_STATUS                    0x8B82
#define GL_INFO_LOG_LENGTH                0x8B84
#define GL_SHADING_LANGUAGE_VERSION       0x8B8C
#define GL_FRAGMENT_SHADER                0x8B30
#define GL_VERTEX_SHADER                  0x8B31
#define GL_RGBA32F_ARB                    0x8814
#define GL_CLAMP_FRAGMENT_COLOR           0x891B
#define GL_FRAMEBUFFER                    0x8D40
#define GL_COLOR_ATTACHMENT0              0x8CE0
#define GL_CLAMP_TO_EDGE                  0x812F

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
PFNGLGENFRAMEBUFFERSPROC glGenFramebuffers;
PFNGLCLAMPCOLORPROC glClampColor;
PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation;
PFNGLUNIFORM1IPROC glUniform1i;
PFNGLUNIFORM4FPROC glUniform4f;
PFNGLUNIFORM1FPROC glUniform1f;
PFNGLBINDFRAMEBUFFEREXTPROC glBindFramebuffer;
PFNGLFRAMEBUFFERTEXTUREPROC glFramebufferTexture;

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
	glGenFramebuffers = (PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers");
	glClampColor = (PFNGLCLAMPCOLORPROC)wglGetProcAddress("glClampColor");
	glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation");
	glUniform1i = (PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i");
	glUniform4f = (PFNGLUNIFORM4FPROC)wglGetProcAddress("glUniform4f");
	glUniform1f = (PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f");	
	glBindFramebuffer =	(PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer");
	glFramebufferTexture = (PFNGLFRAMEBUFFERTEXTUREPROC)wglGetProcAddress("glFramebufferTexture");
}

static const char* VertexShader = \
	"#version 450 \n"	
	"const vec3 vertices[6] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(1,1,0), vec3(-1,1,0)};"
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"gl_Position = vec4(vertices[id], 1);"
	"}";

static const char* BufferA = \
	"#version 450 \n"
	"out vec4 fragColor;"
	"uniform sampler2D iChannel0;"
	"uniform int iFrame;"
	"uniform vec4 iMouse;"
	"const vec2 iResolution = vec2(1280,720);"
	
	"float Circle(vec2 p, vec2 c, float r) "
	"{"
		"return length(p-c)-r;"
	"}"

	"float Map(vec2 p)"
	"{"
		"float sdf = Circle(p, iMouse.xy/iResolution.xy, 0.01);"
		"if (iFrame>0) sdf = min(sdf, texture(iChannel0, p).a); "  
		"return sdf;"
	"}"

	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"
	"{"
		"vec2 p = fragCoord/iResolution.xy;"
		"float sdf = Map(p);"
		"fragColor = vec4(sdf, sdf, sdf, sdf);"
	"}"
	
	"void main()"
	"{"	
		"mainImage(fragColor, gl_FragCoord.xy);"
	"}";
	
static const char* Image = \
	"#version 450 \n"
	"out vec4 fragColor;"
	"uniform sampler2D iChannel0;"
	"uniform float iTime;"
	"uniform vec4 iMouse2;"
	"const vec2 iResolution = vec2(1280,720);"
	
	"float Map(vec2 uv)"
	"{  "
		"return texture(iChannel0, uv).a;"
	"}"

	"float Shadow(vec2 position, vec2 light, float hardness)"
	"{"
		"vec2 direction = normalize(light - position) * 0.5;"
		"float lightDistance = length(light - position) * 0.2;"
		"float rayProgress = 0.0001;"
		"float shadow = 9999.;"
		"for(int i=0; i<32; i++)"
		"{"
			"float sceneDist = Map(position + direction * rayProgress);"
			"if(sceneDist <= 0.) return 0.1;   "     
			"if(rayProgress > lightDistance) return clamp(shadow, 0.1, 1.0);"
			"shadow = min(shadow, hardness * sceneDist / rayProgress);"
			"rayProgress = rayProgress + sceneDist;"
		"}"
		"return 0.1;"
	"}"

	"vec3 Lighting(vec2 p)"
	"{"
		"vec2 lightPos = iMouse2.xy/iResolution.xy;"
		"vec3 lightDir = normalize(vec3(lightPos, -1.0));"
		"vec3 normalDir = normalize(vec3(p, -1.0));"
		"float s = pow(dot(lightDir,normalDir), 5.0); "
		"return vec3(s,s,0) * Shadow(p, lightPos, 50.0);"
	"}"

	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"
	"{"
		"vec2 uv = fragCoord/iResolution.xy;"
		"bool IsPointInsideObject = (Map(uv)<0.0);"
		"fragColor = IsPointInsideObject ? vec4(0,0,1,1) : vec4(Lighting(uv), 1);"
	"}"
	
	"void main()"
	"{"	
		"mainImage(fragColor, gl_FragCoord.xy);"
	"}";
	
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
			free(log);
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
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glInit();
	wglSwapIntervalEXT(0);
	int SHA = LoadShaders(VertexShader, BufferA);
	int SHI = LoadShaders(VertexShader, Image);
	unsigned int FBOA[2];   //BufferA framebuffers
	unsigned int RTA[2];   //BufferA render textures
	glGenFramebuffers(2, FBOA);
	glGenTextures(2, RTA);
	for (int i=0; i<2; i++)
	{
		glBindTexture(GL_TEXTURE_2D, RTA[i]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	glViewport(0, 0, width, height);
	glClampColor( GL_CLAMP_FRAGMENT_COLOR, GL_FALSE );
	int time = glGetUniformLocation(SHI, "iTime");
	int iMouse2 = glGetUniformLocation(SHI, "iMouse2");
	int iChannel0 = glGetUniformLocation(SHA, "iChannel0");
	int iMouse = glGetUniformLocation(SHA, "iMouse");
	int iFrame = glGetUniformLocation(SHA, "iFrame");
	POINT point;	
	DWORD S = GetTickCount();
	bool swap = true;
	int counter = 0;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		GetCursorPos(&point);
		glBindTexture(GL_TEXTURE_2D, RTA[swap]);
		glBindFramebuffer(GL_FRAMEBUFFER, FBOA[!swap]);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, RTA[!swap], 0);
		glUseProgram(SHA);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glUniform1i(iChannel0, 0);
		if (GetAsyncKeyState(VK_LBUTTON)& 0x8000)
			glUniform4f(iMouse, point.x, height - point.y, 0.0f, 0.0f);
		glUniform1i(iFrame, counter);
		counter++;		
		swap = !swap;		
		glBindTexture(GL_TEXTURE_2D, RTA[swap]);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(SHI);
		glUniform1f(time, (GetTickCount()-S)*0.001f);
		if (GetAsyncKeyState(VK_RBUTTON)& 0x8000)
			glUniform4f(iMouse2, point.x, height - point.y, 0.0f, 0.0f);		
		glDrawArrays(GL_TRIANGLES, 0, 6);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}