// g++ -s -o GenerateTexture.exe GenerateTexture.cpp -IGL/inc -LGL/lib -mwindows -lopengl32 -lglew32
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <glew.h>
#include <math.h>

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
"layout (location=1) uniform float time;"
"uniform sampler2D hash;"
"void main()"
"{"
	"vec2 uv = gl_FragCoord.xy/vec2(1920,1080);"
	"color = textureLod(hash,uv,0)*(sin(time)*0.5+0.5);"
"}";

void GenerateNoiseTexture(void)
{
	GLubyte Texels [512][512][4];
	for (int i=0; i<512; i++)
	{
		for (int j=0; j<512; j++)
		{
			float h = fmod(sin(j*12.9898f+i*4.1413f)*43758.5453f,1.0f);
			Texels[i][j][0] = h*255;
			Texels[i][j][1] = h*255;
			Texels[i][j][2] = h*255;
			Texels[i][j][3] = 255;
		}
	}
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,512,512,0,GL_RGBA,GL_UNSIGNED_BYTE,Texels);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
}

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
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);
	GenerateNoiseTexture();
	do 
	{
		GLfloat t = GetTickCount()*0.001f;
		((PFNGLPROGRAMUNIFORM1FVPROC)wglGetProcAddress("glProgramUniform1fv"))( fs, 1, 1, &t);
		glRects(-1,-1,1,1);		
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE); 
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}
