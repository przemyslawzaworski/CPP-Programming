// cl pathtracing1.c opengl32.lib user32.lib gdi32.lib
// Reference: http://iquilezles.org/www/articles/simplepathtracing/simplepathtracing.htm
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdbool.h>

#define width 1920
#define height 1080

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
	"#version 460 \n"	
	"const vec3 vertices[6] = {vec3(-1,-1,0), vec3(1,-1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(1,1,0), vec3(-1,1,0)};"
	"void main()"
	"{"	
		"uint id = gl_VertexID;"
		"gl_Position = vec4(vertices[id], 1);"
	"}";

static const char* BufferA = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"uniform sampler2D iChannel0;"
	"uniform int iFrame;"
	"const vec2 iResolution = vec2(1920,1080);"
	
	"mat3 rotationX( float x )"
	"{"
		"return mat3(1.0,0.0,0.0, 0.0,cos(x),sin(x), 0.0,-sin(x),cos(x));"
	"}"

	"float hash( float seed )"
	"{"
		"uvec2 p = floatBitsToUint(vec2(seed+=.1,seed+=.1));"
		"p = 1103515245U*((p >> 1U)^(p.yx));"
		"uint h32 = 1103515245U*((p.x)^(p.y>>3U));"
		"uint n = h32^(h32 >> 16);"
		"return float(n)/float(0xffffffffU);"
	"}"

	"vec3 hemisphere( float seed, vec3 nor)"
	"{"
		"float u = hash( 78.233 + seed);"
		"float v = hash( 10.873 + seed);"
		"float ks = (nor.z>=0.0) ? 1.0 : -1.0;"
		"float ka = 1.0 / (1.0 + abs(nor.z));"
		"float kb = -ks * nor.x * nor.y * ka;"
		"vec3 uu = vec3(1.0 - nor.x * nor.x * ka, ks*kb, -ks*nor.x);"
		"vec3 vv = vec3(kb, ks - nor.y * nor.y * ka * ks, -nor.y);  "
		"float a = 6.2831853 * v;"
		"return sqrt(u)*(cos(a)*uu + sin(a)*vv) + sqrt(1.0-u)*nor;"
	"}"

	"float map(vec3 p)"
	"{"
		"vec3 q = vec3(mod(p.x, 8.0) - 4.0, p.y, mod(p.z, 8.0) - 4.0);"
		"float cube = length(max(abs(q) - 2.0,0.0));"
		"float sphere = length(q) - 2.5;"
		"return min(p.y + 2.0, max(-sphere, cube));"
	"}"

	"vec3 setNormal( vec3 p )"
	"{"
		"vec3 e = vec3(0.0001,0.0,0.0);"
		"return normalize( vec3(map(p+e.xyy)-map(p-e.xyy), map(p+e.yxy)-map(p-e.yxy), map(p+e.yyx)-map(p-e.yyx)) );"
	"}"

	"float intersect( vec3 ro, vec3 rd )"
	"{"
		"float res = -1.0;"
		"float tmax = 1000.0;"
		"float t = 0.001;"
		"for(int i=0; i<128; i++ )"
		"{"
			"float h = map(ro+rd*t);"
			"if( h<0.0001 || t>tmax ) break;"
			"t += h;"
		"}"
		"if( t<tmax ) res = t;"
		"return res;"
	"}"

	"float shadow( vec3 ro, vec3 rd )"
	"{"
		"float res = 0.0;"
		"float tmax = 100.0;"
		"float t = 0.001;"
		"for(int i=0; i<128; i++)"
		"{"
			"float h = map(ro+rd*t);"
			"if( h<0.0001 || t>tmax) break;"
			"t += h;"
		"}"
		"if( t>tmax ) res = 1.0;"
		"return res;"
	"}"

	"vec3 pathtracer(vec3 ro, vec3 rd, float sa)"
	"{"
		"const float epsilon = 0.0001;"
		"vec3 sunDir = normalize(vec3(200,100.0,-200.0));"
		"vec3 sunCol = 4.0*vec3(1.0,0.95,0.85);"
		"vec3 skyCol = 4.0*vec3(0.2,0.2,0.8);"
		"vec3 colorMask = vec3(1.0);"
		"vec3 accumulatedColor = vec3(0.0);"
		"float fdis = 0.0;"
		"for( int bounce = 0; bounce < 7; bounce++ )"
		"{"
			"float t = intersect( ro, rd );"
			"if( t < 0.0 )"
			"{"
				"if( bounce==0 ) return mix( vec3(1.0), skyCol, smoothstep(0.1,0.25,rd.y) );"
				"break;"
			"}"
			"if( bounce == 0 ) fdis = t;"
			"vec3 pos = ro + rd * t;"
			"vec3 nor = setNormal( pos );"
			"vec3 surfaceColor = vec3(0.05, 0.2, 0.05);"
			"colorMask *= surfaceColor;"
			"vec3 iColor = vec3(0.0);      "
			"float diffuse = max(0.0, dot(sunDir, nor));"
			"float sunSha = 1.0; "
			"if( diffuse > 0.00001 ) sunSha = shadow( pos + nor*epsilon, sunDir);"
			"iColor += sunCol * diffuse * sunSha;"
			"vec3 skyPoint = hemisphere( sa + 7.1*float(iFrame) + 5681.123 + float(bounce)*92.13, nor);"
			"float skySha = shadow( pos + nor*epsilon, skyPoint);"
			"iColor += skyCol * skySha;"
			"accumulatedColor += colorMask * iColor ;"
			"rd = hemisphere(76.2 + 73.1*float(bounce) + sa + 17.7*float(iFrame), nor);"
			"ro = pos;"
		"}"
		"return accumulatedColor;"
	"}"

	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"
	"{"
		"float sa = hash( dot( fragCoord, vec2(12.9898, 78.233) ) + 1113.1*float(iFrame) );"
		"vec2 offset = -0.5 + vec2( hash(sa+13.271), hash(sa+63.216) );"
		"vec2 uv = ( 2.0*(fragCoord+offset) -iResolution.xy) / iResolution.y;"
		"vec3 ro = vec3(0.0, 8.0, 1.0);"
		"vec3 rd = normalize( vec3(uv,2.0) * rotationX(5.7) );"
		"vec3 color = texture( iChannel0, fragCoord/iResolution.xy ).xyz + pathtracer(ro, rd, sa);"
		"fragColor = vec4( color, 1.0 );"
	"}"

	"void main()"
	"{"	
		"mainImage(fragColor, gl_FragCoord.xy);"
	"}";
	
static const char* Image = \
	"#version 460 \n"
	"out vec4 fragColor;"
	"uniform sampler2D iChannel0;"
	"uniform int iFrame;"
	"const vec2 iResolution = vec2(1920,1080);"
	
	"void mainImage( out vec4 fragColor, in vec2 fragCoord )"
	"{"
		"vec2 uv = fragCoord.xy / iResolution.xy;"
		"vec3 pixel = texture( iChannel0, uv ).xyz;"
		"pixel /= float(iFrame);"
		"pixel = pow( pixel, vec3(0.4545) ); "  
		"fragColor = vec4( pixel, 1.0 );"
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
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_POPUP, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glInit();
	wglSwapIntervalEXT(0);
	int SHA = LoadShaders(VertexShader, BufferA);
	int SHI = LoadShaders(VertexShader, Image);
	unsigned int FBOA[2];
	unsigned int RTA[2];
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
	int iChannel0 = glGetUniformLocation(SHA, "iChannel0");
	int iFrame = 0;
	bool swap = true;
	ShowCursor(0);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glBindTexture(GL_TEXTURE_2D, RTA[swap]);
		glBindFramebuffer(GL_FRAMEBUFFER, FBOA[!swap]);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, RTA[!swap], 0);
		glUseProgram(SHA);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glUniform1i(iChannel0, 0);
		glUniform1i(glGetUniformLocation(SHA, "iFrame"), iFrame);		
		swap = !swap;		
		glBindTexture(GL_TEXTURE_2D, RTA[swap]);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(SHI);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
		glUniform1i(glGetUniformLocation(SHI, "iFrame"), iFrame);		
		iFrame++;
	} 
	return 0;
}