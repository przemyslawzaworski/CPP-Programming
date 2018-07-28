// g++ -s -o Demo.exe Demo.cpp -IGL/inc -LGL/lib -lwinmm -mwindows -lopengl32 -lglew32 
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <glew.h>
#include <math.h>
#include <mmsystem.h>
#include <stdio.h>   //only for debug
#include <stdlib.h>   //only for debug

typedef bool (APIENTRY *PFNWGLSWAPINTERVALEXTPROC) (int interval);
static int h[11] = {0x46464952,9172836,0x45564157,0x20746D66,16,WAVE_FORMAT_PCM|131072,44100,176400,1048580,0x61746164,9172800};
static short m[9172822];

static const char* FragmentShader01 = \
"#version 400\n"
"layout (location=0) out vec4 color;"
"uniform float time;"
"uniform sampler2D hash;"
"void main()"
"{"
	"vec2 uv = (2.0*gl_FragCoord.xy-vec2(1366,768))/768;"
	"vec3 o = vec3(0.,0.,-3.);"    
	"for (int i=0;i<64;i++)"
	"{"
		"float l=length(o)-1.;"
		"color=l<.01?dot(vec3(-.6),o)*texture(hash,o.xy+time*.1):color;"
		"o+=l*normalize(vec3(uv,2));"
	"}"
"}";

static const char* FragmentShader02 = \
"#version 400\n"
"layout (location=0) out vec4 color;"
"uniform float time;"
"uniform sampler2D hash;"
"void main()"
"{"
	"vec2 u = gl_FragCoord.xy/37.5;"   
	"u.x+=time*(int(u.y)%2>0?.8:-2.4);"
	"color=texture(hash,floor(u)/256.);"
"}";

int f2i(float x)   //convert float to int, require C++17
{
	if (x>=0x1.0p23) return x;
	return (unsigned int) (x+0.49999997f);
}

float Bass(float t,float n,float e)   //time,15.0,18.0
{
	float x = fmod(t,0.5f);
	return 0.5f * fmax(-1.,fmin(1.,sin(x*pow(2.,((30.-x*5.f-70.)/n))*440.f*6.28f)*10.f*exp(-x*e)));
}

void SetAudio(short *buffer)
{
	for (int i = 0; i<4586400; i++)
	{
		float t = (float)i / (float)44100;		
		float f = Bass(t,15.0,18.0);
		buffer[2*i+0] = f2i(f*32767.0f);
		buffer[2*i+1] = f2i(f*32767.0f);
	}
}

void PlayAudio()
{
	SetAudio(m+22);
	memcpy(m,h,44);
	sndPlaySound((const char*)&m,SND_ASYNC|SND_MEMORY);
}

void GenerateNoiseTexture()
{
	GLubyte Texels [256][256][4];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			float h = fmod(sin(j*12.9898f+i*4.1413f)*43758.5453f,1.0f);
			float k = fmod(cos(i*19.6534f+j*7.9813f)*51364.8733f,1.0f);
			Texels[j][i][0] = h*255;
			Texels[j][i][2] = k*255;
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

void Debug(int sh)
{
	GLint isCompiled = 0;
	((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,GL_LINK_STATUS,&isCompiled);
	if(isCompiled == GL_FALSE)
	{
		GLint length = 0;
		((PFNGLGETSHADERIVPROC)wglGetProcAddress("glGetShaderiv"))(sh,GL_INFO_LOG_LENGTH,&length);
		GLsizei q = 0;
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

int Demo(const char* source)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(GL_FRAGMENT_SHADER);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s,1,&source,0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	Debug(s);
	return p;
}

int main()
{ 
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = {0,0,PFD_DOUBLEBUFFER};
	HDC hdc = GetDC(CreateWindow("static",0,WS_POPUP|WS_VISIBLE|WS_MAXIMIZE,0,0,0,0,0,0,0,0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);	
	int scene1 = Demo(FragmentShader01);
	int scene2 = Demo(FragmentShader02);
	GLint location1 = ((PFNGLGETUNIFORMLOCATIONARBPROC)wglGetProcAddress("glGetUniformLocation"))(scene1,"time");
	GLint location2 = ((PFNGLGETUNIFORMLOCATIONARBPROC)wglGetProcAddress("glGetUniformLocation"))(scene2,"time");	
	GenerateNoiseTexture();
	PlayAudio();
	float S = GetTickCount();   //start timer
	do
	{ 
		GLfloat t = (GetTickCount()- S) * 0.001f;   //update timer
		if (t<8.0f)
		{
			((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(scene1);
			((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location1, 1, &t);
		}
		else
		{
			((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(scene2);
			((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location2, 1, &t);
		}
		glRects(-1,-1,1,1);
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
