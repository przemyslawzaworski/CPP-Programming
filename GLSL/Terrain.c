// gcc -x c -s -o Terrain.exe Terrain.c -mwindows -lopengl32 
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <GL/gl.h>
#include <math.h>

typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char *const *n, const GLint *i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
typedef GLint(APIENTRY* PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY* PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
 
static const char* FragmentShader = \
	"#version 430 \n"
	"layout (location=0) out vec4 color;"
	"uniform float time;"
	"uniform sampler2D hash;"
	"float FBM( vec2 p )"
	"{"
		"float a = 0.5, b = 0.0, t = 0.0;"
		"for (int i=0; i<7; i++)"
		"{"
			"b *= a; t *= a;"
			"b += texture(hash,p).r;"
			"t += 1.0; p /= 2.0;"
		"}"
		"return b /= t;"
	"}"
	"float Map( vec3 p )"
	"{"   
		"if (p.y>2.0) return 1.0;"
		"return ((p.y-2.*FBM(p.xz*0.25))*0.4);"
	"}"
	"bool Raymarch( inout vec3 ro, vec3 rd)"
	"{"
		"float t = 0.0;"
		"for (int i=0; i<256; i++)"
		"{"
			"float d = Map(ro+rd*t);"
			"t+=d;"
			"if (d<t*0.001)"
			"{"
				"ro+=t*rd;"
				"return true;"
			"}"
		"}"
		"return false;"
	"}"
	"vec3 Lighting( vec3 ro, vec3 rd )"
	"{"
		"vec3 c = mix(vec3(0.3,0.6,1),vec3(0.05),rd.y*2.0);"
		"vec3 s = ro;"
		"if (Raymarch(ro,rd))"
		"{"
			"vec3 l = normalize(vec3(-1,1,-1)); "             
			"vec3 m = vec3(0.03,0.06,0.1);"
			"m += clamp(((Map(ro+l*0.05)-Map(ro))/0.05),0.,1.);"   
			"m *= vec3(pow(ro.y,4.)*0.3);"   
			"float f = pow(min(length(ro-s)/18.,1.),2.);"
			"return mix(m, c, f);"
		"}"
		"return c;"
	"}"
	"void main()"
	"{"
		"vec2 uv = (2.*gl_FragCoord.xy - vec2(1920,1080))/1080;"
		"vec3 ro = vec3(0,2,time);"
		"vec3 rd = normalize(vec3(uv,2.0));"  
		"color = vec4(pow(Lighting(ro,rd),vec3(1.0/2.2)),1.0);"
	"}";

void GenerateNoiseTexture()
{
	GLubyte Texels [256][256][4];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			float h = (float)fmod(sin(j*12.9898 + i*4.1413)*43758.5453f, 1.0);
			float k = (float)fmod(cos(i*19.6534 + j*7.9813)*51364.8733f, 1.0);
			Texels[j][i][0] = (GLubyte)(h * 255);
			Texels[j][i][2] = (GLubyte)(k * 255);
		}
	}
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			int x2 = (j - 37) & 255;
			int y2 = (i - 17) & 255;
			Texels[j][i][1] = Texels[x2][y2][0];
			Texels[j][i][3] = Texels[x2][y2][2];
		}
	}
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,256,256,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
}

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	GenerateNoiseTexture();
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s, 1, &FragmentShader, 0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p, s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(p);
	GLint location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(p,"time");
	DWORD S = GetTickCount();	
	do
	{
		float t = (GetTickCount()-S)*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);		
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}