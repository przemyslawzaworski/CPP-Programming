// gcc -x c -s -o PostProcessing.exe PostProcessing.c -mwindows -lopengl32 
#include <windows.h>
#include <GL/gl.h>

typedef int (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint (APIENTRY *PFNGLCREATEPROGRAMPROC)();
typedef GLuint (APIENTRY *PFNGLCREATESHADERPROC)(GLenum t);
typedef void (APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void (APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void (APIENTRY *PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void (APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void (APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void (APIENTRY *PFNGLGENFRAMEBUFFERSPROC) (GLsizei n, GLuint *f);
typedef void (APIENTRY *PFNGLBINDFRAMEBUFFEREXTPROC) (GLenum t, GLuint f);
typedef void (APIENTRY *PFNGLTEXSTORAGE2DPROC) (GLenum t, GLsizei l, GLenum i, GLsizei w, GLsizei h);
typedef void (APIENTRY *PFNGLFRAMEBUFFERTEXTUREPROC) (GLenum t, GLenum a, GLuint s, GLint l);
typedef GLint(APIENTRY *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY *PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);

static const char *PostProcessingShader = \
	"#version 430 \n"
	"layout (location=0) out vec4 color;"
	"uniform sampler2D reader;"
	"void main()"
	"{"	
		"vec3 s = texture(reader,gl_FragCoord.xy/vec2(1920,1080)).rgb;"
		"float grayscale = dot(s, vec3(0.2126, 0.7152, 0.0722));"
		"color = vec4(grayscale,grayscale,grayscale,1.0);"
	"}";
	
static const char *MainShader = \
	"#version 430 \n"
	"layout (location=0) out vec4 color;"
	"uniform float time;"
	"vec3 map( vec3 p )"
	"{"
		"for (int i = 0; i < 30; ++i)"
		"{"
			"p = vec3(1.25,1.07,1.29)*abs(p/dot(p,p)-vec3(0.95,0.91,0.67));"   
		"}"	
		"return p/30.;"
	"}"	
	"vec4 raymarch( vec3 ro, vec3 rd )"
	"{"
		"float T = 3.0;"
		"vec3 c = vec3(0,0,0);"
		"for(int i=0; i<30; ++i)"
		"{"
			"T+=0.1;"
			"c+=map(ro+T*rd);"
		"}"
		"return vec4(c,1.0);"
	"}"	
	"void main()"
	"{"
		"vec2 uv = (2.0*gl_FragCoord.xy-vec2(1920,1080))/1080;"
		"vec3 ro = vec3(0,sin(time*0.2)*5.0,0);"
		"vec3 rd = normalize(vec3(uv,2.0));"
		"color = raymarch(ro,rd);"
	"}";

int MakeShader(const char* source, GLenum type)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(type);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s,1,&source,0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	return p;
}

int main()
{
	ShowCursor(0);
	GLuint framebuffer, texture;
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int MS = MakeShader(MainShader,0x8B30);
	int PS = MakeShader(PostProcessingShader,0x8B30);
	((PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers"))(1, &framebuffer);
	((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, framebuffer);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	((PFNGLTEXSTORAGE2DPROC)wglGetProcAddress("glTexStorage2D"))(GL_TEXTURE_2D, 1, GL_RGBA8, 1920, 1080);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);	
	((PFNGLFRAMEBUFFERTEXTUREPROC)wglGetProcAddress("glFramebufferTexture"))(0x8D40, 0x8CE0, texture, 0);		
	GLint location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(MS,"time");
	DWORD S = GetTickCount();
	do
	{
		((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, framebuffer);
		((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(MS);
		float t = (GetTickCount()-S)*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);		
		glRects(-1, -1, 1, 1);
		((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, 0);
		((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(PS);
		glRects(-1, -1, 1, 1);			
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}