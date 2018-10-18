// cl mandelbulb.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>

typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char *const *n, const GLint *i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
 
static const char* FragmentShader = \
	"#version 450 \n"
	"layout (location=0) out vec4 color;"

	"float map( in vec3 p )"
	"{"
		"vec3 w = p;"
		"float m = dot(w,w);"
		"float dz = 1.0;" 
		"for( int i=0; i<5; i++ )"
		"{"
			"float m2 = m*m;"
			"float m4 = m2*m2;"
			"dz = 8.0*sqrt(m4*m2*m)*dz + 1.0;"
			"float x = w.x; float x2 = x*x; float x4 = x2*x2;"
			"float y = w.y; float y2 = y*y; float y4 = y2*y2;"
			"float z = w.z; float z2 = z*z; float z4 = z2*z2;"
			"float k3 = x2 + z2;"
			"float k2 = inversesqrt( k3*k3*k3*k3*k3*k3*k3 );"
			"float k1 = x4 + y4 + z4 - 6.0*y2*z2 - 6.0*x2*y2 + 2.0*z2*x2;"
			"float k4 = x2 - y2 + z2;"
			"w.x = p.x +  64.0*x*y*z*(x2-z2)*k4*(x4-6.0*x2*z2+z4)*k1*k2;"
			"w.y = p.y + -16.0*y2*k3*k4*k4 + k1*k1;"
			"w.z = p.z +  -8.0*y*k4*(x4*x4 - 28.0*x4*x2*z2+70.0*x4*z4-28.0*x2*z2*z4+z4*z4)*k1*k2;"    
			"m = dot(w,w);"
			"if( m > 256.0 ) break;"
		"}"
		"return 0.25*log(m)*sqrt(m)/dz;"
	"}"

	"vec4 raymarch (vec3 ro, vec3 rd)"
	"{"
		"for (int i=0;i<128;i++)"
		"{"
			"float t = map(ro);"
			"if (t<0.001)"
			"{"
				"float c = pow(1.0-float(i)/float(128),2.0);"
				"return vec4(c,c,c,1.0);"
			"}"
			"ro+=t*rd;"
		"}"
		"return vec4(0,0,0,1);"
	"}"

	"void main()"
	"{"
		"vec2 uv = (2.0*gl_FragCoord.xy - vec2(1920,1080))/1080;"
		"vec3 ro = vec3(0,0.0,-2.5);"
		"vec3 rd = normalize(vec3(uv,2.0));"
		"color = raymarch(ro,rd);"
	"}";

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP|WS_VISIBLE|WS_MAXIMIZE, 0, 0, 19200, 1080, 0, 0, 0, 0));
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
	do
	{	
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}