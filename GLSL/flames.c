// cl flames.c opengl32.lib user32.lib gdi32.lib
// Reference: https://www.shadertoy.com/view/4ttGWM
#include <windows.h>
#include <GL/gl.h>

#define width 1680
#define height 1050

typedef int (WINAPI *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef void (WINAPI *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void (WINAPI *PFNGLDISPATCHCOMPUTEPROC) (GLuint x, GLuint y, GLuint z);
typedef void (WINAPI *PFNGLBINDIMAGETEXTUREPROC) (GLuint a, GLuint b, GLint c, GLboolean d, GLint e, GLenum f, GLenum g);
typedef GLint (WINAPI *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (WINAPI *PFNGLUNIFORM1FPROC) (GLint s, GLfloat v0);
typedef void (WINAPI *PFNGLUNIFORM2IPROC) (GLint s, GLint v0, GLint v1);
typedef GLuint (WINAPI *PFNGLCREATESHADERPROGRAMVPROC) (GLenum t, GLsizei c, const char*const*s);

static const char* ComputeShader = \
	"#version 430 \n"
	"writeonly uniform image2D writer;"
	"uniform float iTime;"
	"uniform ivec2 iResolution;"
	
	"float hash(vec2 n)"
	"{"
		"return fract(sin(cos(dot(n, vec2(12.9898,12.1414)))) * 83758.5453);"
	"}"

	"float noise(vec2 n)"
	"{"
		"vec2 d = vec2(0.0, 1.0);"
		"vec2 b = floor(n), f = smoothstep(vec2(0.0), vec2(1.0), fract(n));"
		"return mix(mix(hash(b), hash(b + d.yx), f.x), mix(hash(b + d.xy), hash(b + d.yy), f.x), f.y);"
	"}"

	"float fbm(vec2 n)"
	"{"
		"float total = 0.0, amplitude = 1.0;"
		"for (int i = 0; i <5; i++)"
		"{"
			"total += noise(n) * amplitude;"
			"n += n*1.7;"
			"amplitude *= 0.47;"
		"}"
		"return total;"
	"}"
	
	"layout (local_size_x = 8, local_size_y = 8) in;"
	"void main()"
	"{"
		"vec2 fragCoord = gl_GlobalInvocationID.xy;"
		"float shift = 1.327+sin(iTime*2.0)/2.4;"
		"vec2 uv = fragCoord.xy / iResolution.xy;"
		"vec2 p = fragCoord.xy * (3.5-sin(iTime*0.4)/1.89) / iResolution.xx;"
		"p += sin(p.yx*4.0+vec2(.2,-.3)*iTime)*0.04;"
		"p += sin(p.yx*8.0+vec2(.6,+.1)*iTime)*0.01;  "
		"p.x -= iTime/1.1;"
		"float q = fbm(p - iTime * 0.3+1.0*sin(iTime+0.5)/2.0);"
		"float qb = fbm(p - iTime * 0.4+0.1*cos(iTime)/2.0);"
		"float q2 = fbm(p - iTime * 0.44 - 5.0*cos(iTime)/2.0) - 6.0;"
		"float q3 = fbm(p - iTime * 0.9 - 10.0*cos(iTime)/15.0)-4.0;"
		"float q4 = fbm(p - iTime * 1.4 - 20.0*sin(iTime)/14.0)+2.0;"
		"q = (q + qb - .4 * q2 -2.0*q3  + .6*q4)/3.8;"
		"vec2 r = vec2(fbm(p + q /2.0 + iTime * 0.1 - p.x - p.y), fbm(p + q - iTime * 0.9));"
		"vec3 color = vec3(1.0,.2,.05)/(pow((r.y+r.y)* max(.0,p.y)+0.1, 4.0));"
		"color = color/(1.0+max(vec3(0),color));"
		"vec4 fragColor = vec4(color, 1.0);"
		"imageStore(writer,ivec2(gl_GlobalInvocationID),fragColor);"
	"}";

static LRESULT CALLBACK WindowProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	if( uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE) )
	{
		PostQuitMessage(0); return 0;
	}
	if( uMsg==WM_SIZE )
	{
		glViewport( 0, 0, lParam&65535, lParam>>16 );
	}
	return(DefWindowProc(hWnd,uMsg,wParam,lParam));
} 

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Compute Shader"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Compute Shader", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);	
	glOrtho(0, width, 0, height, -1, 1);
	int CS = ((PFNGLCREATESHADERPROGRAMVPROC)wglGetProcAddress("glCreateShaderProgramv")) (0x91B9, 1, &ComputeShader);
	glBindTexture(GL_TEXTURE_2D, 1);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,width,height,0,GL_RGBA,GL_FLOAT,0);
	((PFNGLBINDIMAGETEXTUREPROC)wglGetProcAddress("glBindImageTexture"))(0,1,0,GL_FALSE,0,0x88B9,GL_RGBA8);
	glEnable(GL_TEXTURE_2D);
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