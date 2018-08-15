// gcc 4klang.obj -x c -s -o Audio.exe  Audio.c -lwinmm -mwindows -lopengl32
// Code works also with Visual Studio and Crinkler (only change entrypoint and that's it) 
// Install Renoise, add plugin 4klang.dll, open *.xrns file to show 4klang GUI.
// In Renoise main window press play, and then press "record" in 4klang window.
// Then press "stop", and save to HDD. It will generate new 4klang.h
// (you can replace lines 12-32 from Audio.c with new values from 4klang.h) and 4klang.obj
// Compile with commands from line 1. Play.
#include <windows.h>
#include <GL/gl.h>
#include <mmsystem.h>
#include <mmreg.h>

#define SAMPLE_RATE 44100
#define BPM 125.000000
#define MAX_INSTRUMENTS 1
#define MAX_PATTERNS 25
#define PATTERN_SIZE_SHIFT 4
#define PATTERN_SIZE (1 << PATTERN_SIZE_SHIFT)
#define MAX_TICKS (MAX_PATTERNS*PATTERN_SIZE)
#define SAMPLES_PER_TICK 5292
#define MAX_SAMPLES (SAMPLES_PER_TICK*MAX_TICKS)
#define POLYPHONY 1
#define FLOAT_32BIT
#define SAMPLE_TYPE float
#define WINDOWS_OBJECT

#ifdef __cplusplus
extern "C" {
#endif
void  __stdcall	_4klang_render(void*);
#ifdef __cplusplus
}
#endif

typedef int (APIENTRY *PFNWGLSWAPINTERVALEXTPROC)(int i);
typedef GLuint(APIENTRY *PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY *PFNGLCREATESHADERPROC)(GLenum t);
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


SAMPLE_TYPE	lpSoundBuffer[MAX_SAMPLES * 2];
HWAVEOUT hWaveOut;
WAVEHDR WaveHDR = {(LPSTR)lpSoundBuffer,MAX_SAMPLES * sizeof(SAMPLE_TYPE) * 2,0,0,0,0,0,0};
MMTIME MMTime = { TIME_SAMPLES,0 };
WAVEFORMATEX WaveFMT =
{	
	WAVE_FORMAT_IEEE_FLOAT,2,SAMPLE_RATE,SAMPLE_RATE*sizeof(SAMPLE_TYPE)*2,sizeof(SAMPLE_TYPE)*2,sizeof(SAMPLE_TYPE)*8,0 
};

void MakeSound()
{
	CreateThread(0, 0, (LPTHREAD_START_ROUTINE)_4klang_render, lpSoundBuffer, 0, 0);
	waveOutOpen(&hWaveOut, WAVE_MAPPER, &WaveFMT, (DWORD)NULL, 0, CALLBACK_NULL);
	waveOutPrepareHeader(hWaveOut, &WaveHDR, sizeof(WaveHDR));
	waveOutWrite(hWaveOut, &WaveHDR, sizeof(WaveHDR));
	waveOutGetPosition(hWaveOut, &MMTime, sizeof(MMTIME));
}

int MakeShader(const char* source, GLenum type)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(type);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s, 1, &source, 0);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p, s);
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	return p;
}

int main()
{
	ShowCursor(0);
	GLuint framebuffer, texture;
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP | WS_VISIBLE | WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	int MS = MakeShader(MainShader, 0x8B30);
	int PS = MakeShader(PostProcessingShader, 0x8B30);
	((PFNGLGENFRAMEBUFFERSPROC)wglGetProcAddress("glGenFramebuffers"))(1, &framebuffer);
	((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, framebuffer);
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	((PFNGLTEXSTORAGE2DPROC)wglGetProcAddress("glTexStorage2D"))(GL_TEXTURE_2D, 1, GL_RGBA8, 1920, 1080);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	((PFNGLFRAMEBUFFERTEXTUREPROC)wglGetProcAddress("glFramebufferTexture"))(0x8D40, 0x8CE0, texture, 0);
	GLint location = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(MS, "time");
	DWORD S = GetTickCount();
	MakeSound();
	do
	{
		((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, framebuffer);
		((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(MS);
		float t = (GetTickCount() - S)*0.001f;
		((PFNGLUNIFORM1FVPROC)wglGetProcAddress("glUniform1fv"))(location, 1, &t);
		glRects(-1, -1, 1, 1);
		((PFNGLBINDFRAMEBUFFEREXTPROC)wglGetProcAddress("glBindFramebuffer"))(0x8D40, 0);
		((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(PS);
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}