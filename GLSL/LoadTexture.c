// gcc -x c -s -o LoadTexture.exe LoadTexture.c -mwindows -lopengl32
// Texture loader is compatible with GIMP PPM (PNM) binary file format.
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdlib.h>

typedef int(APIENTRY *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY *PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY *PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY *PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char *const *n, const GLint *i);
typedef void(APIENTRY *PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY *PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY *PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef GLint(APIENTRY *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY *PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
typedef void (APIENTRY *PFNGLUNIFORM1IPROC) (GLint l, GLint v);
typedef void (APIENTRY *PFNGLACTIVETEXTUREPROC) (GLenum t);

static const char* FragmentShader = \
	"#version 400 \n"
	"layout (location=0) out vec4 color;"
	"uniform float time;"
	"uniform sampler2D clouds;"
	"uniform sampler2D ground;"
	"void main()"
	"{"
		"vec3 d = normalize(vec3((2.0*gl_FragCoord.xy-vec2(1366,768))/768,2.0));"
		"vec3 p = d*(-0.2/d.y);"
		"p.z+=time*0.1;"
		"color = max(texture(clouds,p.xz)*(3.0*d.y),texture(ground,p.xz)*(-3.0*d.y));"		
	"}";

void LoadTexture(char *filename, int unit, int id, int shader, char *name)
{
	int width, height; 
	char buffer[128]; 
	FILE *file = fopen(filename, "rb");
	fgets(buffer, sizeof(buffer), file);
	do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
	sscanf (buffer, "%d %d", &width, &height);
	do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
	int size = width * height * 4 * sizeof(GLubyte);
	GLubyte *Texels  = (GLubyte *)malloc(size);
	for (int i = 0; i < size; i++) 
	{
		Texels[i] = ((i % 4) < 3 ) ? (GLubyte) fgetc(file) : (GLubyte) 255 ;
	}
	fclose(file);
	((PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture"))(unit);
	glBindTexture(GL_TEXTURE_2D, id);	
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	int num = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(shader, name);
	((PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i"))(num, id);		
}

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc)); 
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s, 1, &FragmentShader, 0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p, s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(p);
	LoadTexture("clouds.ppm", 0x84C0, 0, p, "clouds");
	LoadTexture("plaster.ppm", 0x84C1, 1, p, "ground");
	int location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(p,"time");	
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