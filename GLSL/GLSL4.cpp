// g++ -s -o GLSL4.exe GLSL4.cpp -IGL/inc -LGL/lib -mwindows -lopengl32 -lglew32
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <glew.h>

typedef bool (APIENTRY *PFNWGLSWAPINTERVALEXTPROC) (int interval);

static const char* VertexShader = \
"#version 450\n"
"layout (location=0) in vec3 position;"
"void main()"
"{"
	"gl_Position=vec4(position,1.0);"
"}";

static const char* FragmentShader = \
"#version 450\n"
"layout (location=0) out vec4 color;"
"void main()"
"{"
	"vec2 uv = gl_FragCoord.xy/vec2(1920,1080);"
	"color = vec4(uv,0,1);"
"}";

int main()
{
	ShowCursor(0);
	PIXELFORMATDESCRIPTOR pfd = {0,0,PFD_DOUBLEBUFFER};
	HDC hdc = GetDC(CreateWindow((LPCSTR)0xC018,0,WS_POPUP|WS_VISIBLE|WS_MAXIMIZE,0,0,1920,1080,0,0,0,0));
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	int vs = ((PFNGLCREATESHADERPROGRAMVPROC)wglGetProcAddress("glCreateShaderProgramv")) (GL_VERTEX_SHADER, 1, &VertexShader);
	int fs = ((PFNGLCREATESHADERPROGRAMVPROC)wglGetProcAddress("glCreateShaderProgramv")) (GL_FRAGMENT_SHADER, 1, &FragmentShader);
	unsigned int p;
	((PFNGLGENPROGRAMPIPELINESPROC)wglGetProcAddress("glGenProgramPipelines")) (1, &p);
	((PFNGLBINDPROGRAMPIPELINEPROC)wglGetProcAddress("glBindProgramPipeline")) (p);
	((PFNGLUSEPROGRAMSTAGESPROC)wglGetProcAddress("glUseProgramStages"))(p, GL_VERTEX_SHADER_BIT, vs);
	((PFNGLUSEPROGRAMSTAGESPROC)wglGetProcAddress("glUseProgramStages"))(p, GL_FRAGMENT_SHADER_BIT, fs);
	((PFNWGLSWAPINTERVALEXTPROC) wglGetProcAddress ("wglSwapIntervalEXT")) (0);
	do 
	{
		glRects(-1,-1,1,1);		
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE); 
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
