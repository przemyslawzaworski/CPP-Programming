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
"void main()"
"{"
	"vec2 uv = (2.0*gl_FragCoord.xy-vec2(1366,768))/768;"
	"vec3 p = vec3(0.0,0.0,-3.0);"    
	"for (int i=0;i<128;i++)"
	"{"
		"float d=length(p)-1.0;"
		"p+=d*normalize(vec3(uv,2.0));"		
		"color=d<.01?dot(vec3(-0.6),p)*texture(reader,p.xy+time*0.1):color;"
	"}"
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