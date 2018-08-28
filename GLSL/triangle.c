#include <windows.h>
#include <GL/gl.h>
#include <stddef.h>

typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void (APIENTRY* PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void (APIENTRY* PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void (APIENTRY* PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef void (APIENTRY* PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void (APIENTRY* PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void (APIENTRY* PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void (APIENTRY* PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef int (APIENTRY* PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef GLint(APIENTRY *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY *PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
typedef GLint (APIENTRY* PFNGLGETATTRIBLOCATIONPROC) (GLuint p, const char *n);
typedef void (APIENTRY* PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);

static const GLfloat vertices[] = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
static const GLfloat colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
unsigned int VertexBuffer, ColorBuffer, VertexArrayID;

static const char* VertexShader = \
	"#version 430 core\n"
	"layout (location=0) in vec3 vertex;"
	"layout (location=1) in vec3 vertexColor;"
	"out vec3 fragmentColor;"
	"uniform float time;"
	"void main()"
	"{"	
		"vec3 position = vertex * sin(time);"
		"fragmentColor = vertexColor;"
		"gl_Position = vec4(position,1.0);"

	"}";
	
static const char* FragmentShader = \
	"#version 430 core\n"
	"layout (location=0) out vec4 color;"
	"layout (location=1) in vec3 fragmentColor;"
	"void main()"
	"{"	
		"color = vec4(fragmentColor,1);"
	"}";
	
int MakeShader(const char* VS, const char* FS)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s1 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B31);	
	int s2 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s1,1,&VS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s2,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s1);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s2);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s1);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s2);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	return p;
}

int main()
{
	PIXELFORMATDESCRIPTOR pfd = { 0, 0, PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP | WS_VISIBLE | WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	ShowCursor(0);
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	((PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays")) (1, &VertexArrayID);		
	((PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray")) (VertexArrayID);	
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &VertexBuffer);
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(vertices), vertices, 0x88E4);
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &ColorBuffer);	
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, ColorBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(colors), colors, 0x88E4);		
	int PS = MakeShader(VertexShader,FragmentShader);	
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(PS);
	int location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(PS,"time");	
	DWORD S = GetTickCount();	
	do
	{
		glClear(GL_COLOR_BUFFER_BIT);
		float t = (GetTickCount()-S)*0.001f;	
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(0);
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );		
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(1);	
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, ColorBuffer);		
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(1,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );		
		glDrawArrays(GL_TRIANGLES, 0, 3);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(0);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(1);		
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}