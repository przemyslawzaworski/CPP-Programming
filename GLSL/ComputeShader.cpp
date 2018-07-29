// g++ -s -o ComputeShader.exe ComputeShader.cpp -mwindows -lopengl32  
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <GL/gl.h>
#include "glext.h"
#include <stdio.h>   //only for debug
#include <stdlib.h>   //only for debug

typedef bool (APIENTRY *PFNWGLSWAPINTERVALEXTPROC) (int interval);

static const char* ComputeShader = \
"#version 430 core\n"
"writeonly uniform image2D writer;"
"layout (local_size_x = 16, local_size_y = 16) in;"
"void main()"
"{"
	"vec2 coordinates = gl_GlobalInvocationID.xy;"
	"vec2 resolution = vec2(512,512);"
	"vec2 k = sign(cos(coordinates/resolution.yy*32.0));"
	"imageStore(writer,ivec2(gl_GlobalInvocationID.xy),vec4(k.x*k.y));"
"}" ;
	
static const char* FragmentShader = \
"#version 430 core\n"
"layout (location=0) out vec4 color;"
"uniform float time;"
"uniform sampler2D reader;"
"mat3 rotationX(float x)"
"{"
	"return mat3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));"
"}"
"mat3 rotationY(float y)" 
"{"
	"return mat3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));"
"}"
"float Cuboid (vec3 p,vec3 c,vec3 s)"
"{"
	"vec3 d = abs(p-c)-s;"
	"return max(max(d.x,d.y),d.z);"
"}"
"float Map (vec3 p)"
"{"
	"return Cuboid(p,vec3(0.0,0.0,0.0),vec3(2.0,2.0,2.0));"
"}"
"vec4 SetTexture (sampler2D s, vec3 pos, vec3 nor)"
"{"
	"vec3 w = nor*nor;"
	"return (w.x*texture(s,pos.yz)+w.y*texture(s,pos.zx)+w.z*texture(s,pos.xy))/(w.x+w.y+w.z);" 
"}"
"vec3 SetNormal (vec3 p)"
"{"
	"vec2 e = vec2(0.01,0.00);"
	"return normalize(vec3(Map(p+e.xyy)-Map(p-e.xyy),Map(p+e.yxy)-Map(p-e.yxy),Map(p+e.yyx)-Map(p-e.yyx)));"
"}"
"vec4 Raymarch (vec3 ro, vec3 rd)"
"{"
	"for (int i=0; i<128; i++)"
	"{"
		"float t = Map(ro);"
		"if (t<0.001) return SetTexture(reader,ro*0.2,SetNormal(ro));"
		"ro+=t*rd;"
	"}"
	"return vec4(0.0,0.0,0.5,0.0);"
"}"
"void main()"
"{"
	"vec2 uv = (2.0*gl_FragCoord.xy-vec2(1366,768))/768;"	
	"vec3 ro = vec3(0.0,0.0,-10.0);"
	"vec3 rd = normalize(vec3(uv,2.0));"
	"ro*=rotationY(time)*rotationX(time);"
	"rd*=rotationY(time)*rotationX(time);"
	"color = Raymarch(ro,rd);"
"}";

void Debug(int sh)
{
	int isCompiled = 0;
	((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,GL_LINK_STATUS,&isCompiled);
	if(isCompiled == GL_FALSE)
	{
		int length = 0;
		((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,GL_INFO_LOG_LENGTH,&length);
		int q = 0;
		char* log = (char*)malloc(sizeof(char)*length);
		((PFNGLGETSHADERINFOLOGPROC)wglGetProcAddress("glGetShaderInfoLog"))(sh,length,&q,log);
		if (length>1)
		{
			FILE *file = fopen ("debug.log","a");
			fprintf (file,"%s\n%s\n",(char*)glGetString(GL_SHADING_LANGUAGE_VERSION),log);
			fclose (file);
			ExitProcess(0);
		}
	}
}

void GenerateTexture() 
{
	GLuint h;
	glBindTexture(GL_TEXTURE_2D, h);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,512,512,0,GL_RGBA,GL_FLOAT,0);
	((PFNGLBINDIMAGETEXTUREPROC)wglGetProcAddress("glBindImageTexture"))(0,h,0,GL_FALSE,0,GL_WRITE_ONLY,GL_RGBA8);
}

int MakeShader(const char* source, GLenum type)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(type);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s,1,&source,0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(s); return p;
}

int main()
{ 
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = {0,0,PFD_DOUBLEBUFFER};
	HDC hdc = GetDC(CreateWindow("static",0,WS_POPUP|WS_VISIBLE|WS_MAXIMIZE,0,0,0,0,0,0,0,0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);	
	int FS = MakeShader(FragmentShader,GL_FRAGMENT_SHADER);
	int CS = MakeShader(ComputeShader,GL_COMPUTE_SHADER);
	GenerateTexture();
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(CS);
	((PFNGLDISPATCHCOMPUTEPROC)wglGetProcAddress("glDispatchCompute"))(512/16,512/16,1);
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(FS);				
	GLint location = ((PFNGLGETUNIFORMLOCATIONARBPROC)wglGetProcAddress("glGetUniformLocation"))(FS,"time");
	float S = GetTickCount();   //start timer	
	do
	{ 
		float t = (GetTickCount()-S)*0.001f;   //update timer
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);
		glRects(-1,-1,1,1);
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
