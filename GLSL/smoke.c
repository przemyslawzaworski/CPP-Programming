// cl smoke.c opengl32.lib user32.lib gdi32.lib
// Przemyslaw Zaworski 11.12.2018

#include <windows.h>
#include <GL/gl.h>
#include <math.h>

typedef int(__stdcall *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(__stdcall *PFNGLCREATEPROGRAMPROC)();
typedef GLuint(__stdcall *PFNGLCREATESHADERPROC)(GLenum t);
typedef void(__stdcall *PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char *const *n, const GLint *i);
typedef void(__stdcall *PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(__stdcall *PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(__stdcall *PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(__stdcall *PFNGLUSEPROGRAMPROC)(GLuint p);
typedef GLint(__stdcall *PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void (__stdcall *PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
typedef void(__stdcall *PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void(__stdcall *PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void(__stdcall *PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void(__stdcall *PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void(__stdcall *PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);
typedef void(__stdcall *PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void(__stdcall *PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void(__stdcall *PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef void(__stdcall *PFNGLACTIVETEXTUREPROC) (GLenum t);
typedef void(__stdcall *PFNGLUNIFORM1IPROC) (GLint l, GLint v);

static const GLfloat vertices[] = {-1.0f,-1.0f,0.0f,1.0f,-1.0f,0.0f,-1.0f,1.0f,0.0f,1.0f,-1.0f,0.0f,1.0f,1.0f,0.0f,-1.0f,1.0f,0.0f};

static const char* FragmentShader = \
	"#version 430 \n"
	"layout (location=0) out vec4 color;"
	"uniform float time;"
	"uniform sampler2D hash;"
	"mat3 rotationX(float x)"
	"{"
		"return mat3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));"
	"}"
	"float noise( in vec3 x )"
	"{"
		"vec3 p = floor(x);"
		"vec3 f = fract(x);"
		"f = f*f*(3.0-2.0*f);"
		"vec2 uv = (p.xy+vec2(37.0,17.0)*p.z) + f.xy;"
		"vec2 rg = textureLod( hash, (uv+ 0.5)/256.0, 0.0 ).yx;"
		"return mix( rg.x, rg.y, f.z );"
	"}"
	"vec4 map( vec3 p )"
	"{"
		"float d = 0.2 - p.y;"	
		"vec3 q = p  - vec3(0.0,1.0,0.0)*time;"
		"float f  = 0.50000*noise( q ); q = q*2.02 - vec3(0.0,1.0,0.0)*time;"
		"f += 0.25000*noise( q ); q = q*2.03 - vec3(0.0,1.0,0.0)*time;"
		"f += 0.12500*noise( q ); q = q*2.01 - vec3(0.0,1.0,0.0)*time;"
		"f += 0.06250*noise( q ); q = q*2.02 - vec3(0.0,1.0,0.0)*time;"
		"f += 0.03125*noise( q );"
		"d = clamp( d + 4.5*f, 0.0, 1.0 );"
		"vec3 col = mix( vec3(1.0,0.9,0.8), vec3(0.4,0.1,0.1), d ) + 0.05*sin(p);"
		"return vec4( col, d );"
	"}"
	"vec3 raymarch( vec3 ro, vec3 rd )"
	"{"
		"vec4 s = vec4( 0,0,0,0 );"
		"float t = 0.0;"	
		"for( int i=0; i<128; i++ )"
		"{"
			"if( s.a > 0.99 ) break;"
			"vec3 p = ro + t*rd;"
			"vec4 k = map( p );"
			"k.rgb *= mix( vec3(3.0,1.5,0.15), vec3(0.5,0.5,0.5), clamp( (p.y-0.2)/2.0, 0.0, 1.0 ) );"
			"k.a *= 0.5;"
			"k.rgb *= k.a;"
			"s = s + k*(1.0-s.a);"	
			"t += 0.05;"
		"}"
		"return clamp( s.xyz, 0.0, 1.0 );"
	"}"
	"void main()"
	"{"
		"vec3 ro = vec3(0.0,4.9,-40.);"
		"vec3 rd = normalize(vec3((2.0*gl_FragCoord.xy-vec2(1280,720))/720,2.0)) * rotationX(5.2);"
		"vec3 volume = raymarch( ro, rd );"
		"volume = volume*0.5 + 0.5*volume*volume*(3.0-2.0*volume);"
		"color = vec4( volume, 1.0 );"
	"}";

void SetTexture (int shader, const char *name)
{
	unsigned char Texels [256][256][4];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			float h = (float)fmod(sin(j*12.9898 + i*4.1413)*43758.5453f, 1.0);
			float k = (float)fmod(cos(i*19.6534 + j*7.9813)*51364.8733f, 1.0);
			Texels[j][i][0] = (unsigned char)(h * 255);
			Texels[j][i][2] = (unsigned char)(k * 255);
		}
	}
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			int x2 = (j - 17) & 255;
			int y2 = (i - 37) & 255;
			Texels[j][i][1] = Texels[x2][y2][0];
			Texels[j][i][3] = Texels[x2][y2][2];
		}
	}
	((PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture"))(0x84C0);
	glBindTexture(0x806F, 0);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,256,256,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	int loc = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(shader, name);
	((PFNGLUNIFORM1IPROC)wglGetProcAddress("glUniform1i"))(loc, 0);	
}

static LRESULT CALLBACK WindowProc( HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam )
{
	if( uMsg==WM_SYSCOMMAND && (wParam==SC_SCREENSAVE || wParam==SC_MONITORPOWER) )
		return 0;
	if( uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE) )
	{
		PostQuitMessage(0);
		return 0;
	}
	if( uMsg==WM_SIZE )
	{
		glViewport( 0, 0, lParam&65535, lParam>>16 );
	}
	if( uMsg==WM_CHAR || uMsg==WM_KEYDOWN)
	{
		if( wParam==VK_ESCAPE )
		{
			PostQuitMessage(0);
			return 0;
		}
	}
	return(DefWindowProc(hWnd,uMsg,wParam,lParam));
} 

int WINAPI WinMain( HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow )
{
	MSG msg;
	int exit = 0;
	unsigned int VertexBuffer, VertexArrayID;
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	WNDCLASS win;
	ZeroMemory( &win, sizeof(WNDCLASS) );
	win.style = CS_OWNDC|CS_HREDRAW|CS_VREDRAW;
	win.lpfnWndProc = WindowProc;
	win.hInstance = 0;
	win.lpszClassName = "smoke";
	win.hbrBackground =(HBRUSH)(COLOR_WINDOW+1);
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Volumetric Smoke", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, 1280, 720, 0, 0, 0, 0));
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	((PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays")) (1, &VertexArrayID);
	((PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray")) (VertexArrayID);
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &VertexBuffer);
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(vertices), vertices, 0x88E4);
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s, 1, &FragmentShader, 0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p, s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	SetTexture (p, "hash");	
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(p);
	GLint location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(p,"time");
	DWORD S = GetTickCount();
	while( !exit )
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE) )
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float t = (GetTickCount()-S)*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(0);
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );
		glDrawArrays(GL_TRIANGLES, 0, 2*3);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(0);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	}
	return 0;
}