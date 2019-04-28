// cl stain.c opengl32.lib user32.lib gdi32.lib

#include <windows.h>
#include <GL/gl.h>

#define width 1280
#define height 720

typedef int (WINAPI *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint (WINAPI *PFNGLCREATEPROGRAMPROC)();
typedef GLuint (WINAPI *PFNGLCREATESHADERPROC)(GLenum t);
typedef void (WINAPI *PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void (WINAPI *PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void (WINAPI *PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void (WINAPI *PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void (WINAPI *PFNGLDISPATCHCOMPUTEPROC) (GLuint x, GLuint y, GLuint z);
typedef void (WINAPI *PFNGLBINDIMAGETEXTUREPROC) (GLuint a, GLuint b, GLint c, GLboolean d, GLint e, GLenum f, GLenum g);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (WINAPI *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);
typedef void (WINAPI *PFNGLUNIFORM2IPROC) (GLint s, GLint v0, GLint v1);

static const char* ComputeShader = \
	"#version 430 \n"
	"writeonly uniform image2D writer;"
	"uniform float iTime;"
	"uniform ivec2 iResolution;"
	"layout (local_size_x = 8, local_size_y = 8) in;"
	"void main()"
	"{"
		"vec2 fragCoord = gl_GlobalInvocationID.xy;"
		"vec2 uv = (2.0 * fragCoord.xy - iResolution.xy) / iResolution.y;"
		"vec3 k = sin(vec3(iTime*.08, iTime*.03, iTime*.007));"
		"for (int i = 0; i < 30; i++) "
		"{"
			"vec3 p = vec3(uv*float(i),float(i));"
			"k += vec3( cos(k.y+sin(p.x)), sin(k.z+cos(p.z)), -sin(k.x+sin(p.y)) );"
		"}"
		"vec4 fragColor = vec4(k*0.06,1.0);"
		"imageStore(writer,ivec2(gl_GlobalInvocationID),fragColor);"
	"}" ;

void MakeBuffer() 
{
	glBindTexture(GL_TEXTURE_2D, 1);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_FLOAT,0);
	((PFNGLBINDIMAGETEXTUREPROC)wglGetProcAddress("glBindImageTexture"))(0,1,0,GL_FALSE,0,0x88B9,GL_RGBA8);
	glEnable(GL_TEXTURE_2D);
}

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

static LRESULT CALLBACK WindowProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	if( uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE) )
	{
		PostQuitMessage(0);
		return 0;
	}
	if( uMsg==WM_SIZE )
	{
		glViewport( 0, 0, lParam&65535, lParam>>16 );
	}
	return(DefWindowProc(hWnd,uMsg,wParam,lParam));
} 

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
	MSG msg;
	int exit = 0;
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	WNDCLASS win;
	ZeroMemory( &win, sizeof(WNDCLASS) );
	win.style = CS_OWNDC|CS_HREDRAW|CS_VREDRAW;
	win.lpfnWndProc = WindowProc;
	win.hInstance = 0;
	win.lpszClassName = "noname";
	win.hbrBackground =(HBRUSH)(COLOR_WINDOW+1);
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Compute Shader", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);	
	glOrtho(0, width, 0, height, -1, 1);
	int CS = MakeShader(ComputeShader,0x91B9);
	MakeBuffer();
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(CS);
	int iTime = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(CS,"iTime");
	GLint iResolution = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(CS,"iResolution");
	((PFNGLUNIFORM2IPROC)wglGetProcAddress("glUniform2i"))(iResolution, width, height);	
	while( !exit )
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE) )
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glBindTexture(GL_TEXTURE_2D, 1);
		((PFNGLUNIFORM1FPROC)wglGetProcAddress("glUniform1f"))(iTime, GetTickCount()*0.001f);
		((PFNGLDISPATCHCOMPUTEPROC)wglGetProcAddress("glDispatchCompute"))(width/8, height/8, 1);		
		glBegin(GL_QUADS);
		glTexCoord2i(0, 0); glVertex2i(0, 0);
		glTexCoord2i(0, 1); glVertex2i(0, height);
		glTexCoord2i(1, 1); glVertex2i(width, height);
		glTexCoord2i(1, 0); glVertex2i(width, 0);
		glEnd();
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE);
	} 
	return 0;
}