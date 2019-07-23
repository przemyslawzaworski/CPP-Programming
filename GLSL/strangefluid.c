// cl strangefluid.c opengl32.lib user32.lib gdi32.lib
// Reference: https://www.shadertoy.com/view/XdcGW2
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdbool.h>

#define width 800
#define height 450

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
	"uniform float iTime;"
	"uniform vec4 iMouse;"
	"const vec2 iResolution = vec2(800,450);"
	
	"vec4 get_pixel(float x_offset, float y_offset)"
	"{"
		"return texture(iChannel0, (gl_FragCoord.xy/ iResolution.xy) + (vec2(x_offset, y_offset) / iResolution.xy));"
	"}" 

	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"	
	"{"
		"float val = get_pixel(0.0, 0.0).r;"    
		"val += fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453)*val*0.15;"
		"val = get_pixel(sin(get_pixel(val, 0.0).r  - get_pixel(-val, 0.0) + 3.1415).r  * val * 0.4,"
			"cos(get_pixel(0.0, -val).r - get_pixel(0.0 , val) - 3.1415/2.0).r * val * 0.4).r;"   
		"val *= 1.0001;"
		"if(iTime < 0.05)" 
			"val = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898,78.233))) * 43758.5453)*length(iResolution.xy)/100.0 +"
			"smoothstep(length(iResolution.xy)/2.0, 0.5, length(iResolution.xy * 0.5 - gl_FragCoord.xy))*25.0;" 
		"if (iMouse.z > 0.0) val += smoothstep(length(iResolution.xy)/10.0, 0.5, length(iMouse.xy - fragCoord.xy));"
		"fragColor = vec4(val,0,0,1);"
	"}"
	
	"void main()"
	"{"	
		"mainImage(fragColor, gl_FragCoord.xy);"
	"}";
	
static const char* Image = \
	"#version 450 \n"
	"out vec4 fragColor;"
	"uniform sampler2D iChannel0;"
	"const vec2 iResolution = vec2(800,450);"
	
	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"
	"{"	
		"float val = texture(iChannel0, gl_FragCoord.xy/ iResolution.xy).r;"
		"vec4 color = pow(vec4(cos(val), tan(val), sin(val), 1.0) * 0.5 + 0.5, vec4(0.5));"      
		"vec2 q = gl_FragCoord.xy/ iResolution.xy;"  
		"vec3 e = vec3(vec2(1.0)/iResolution.xy,0.);"
		"float p10 = texture(iChannel0, q-e.zy).x;"
		"float p01 = texture(iChannel0, q-e.xz).x;"
		"float p21 = texture(iChannel0, q+e.xz).x;"
		"float p12 = texture(iChannel0, q+e.zy).x; "      
		"vec3 grad = normalize(vec3(p21 - p01, p12 - p10, 1.));"
		"vec3 light = normalize(vec3(.2,-.25,.7));"
		"float diffuse = dot(grad, light);"
		"float spec = pow(max(0.,-reflect(light,grad).z),32.0);  " 
		"fragColor = (color * diffuse) + spec;"
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
	wglSwapIntervalEXT(1);
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
	int time = glGetUniformLocation(SHA, "iTime");
	int iChannel0 = glGetUniformLocation(SHA, "iChannel0");
	int iMouse = glGetUniformLocation(SHA, "iMouse");
	POINT point;	
	DWORD S = GetTickCount();
	bool swap = true;
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
		glUniform1f(time, (GetTickCount()-S)*0.001f);
		glUniform4f(iMouse, point.x, height - point.y, GetAsyncKeyState(VK_LBUTTON)& 0x8000, GetAsyncKeyState(VK_RBUTTON)& 0x8000);
		
		swap = !swap;
		
		glBindTexture(GL_TEXTURE_2D, RTA[swap]);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(SHI);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}