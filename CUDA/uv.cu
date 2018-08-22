// nvcc -o uv.exe uv.cu -IE:\CUDA\Samples\common\inc -lopengl32 -arch=sm_30  user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>
#include <helper_math.h> 
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define width 1920
#define height 1080
static float timer = 0.0f;
float3* device;
unsigned int buffer;
typedef void (APIENTRY* PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void (APIENTRY* PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void (APIENTRY* PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef int (APIENTRY* PFNWGLSWAPINTERVALEXTPROC) (int i);

__global__ void rendering(float3 *output, float k)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;
	float2 resolution = make_float2((float)width, (float)height);
	float2 coordinates = make_float2((float)x, (float)y);
	float2 uv = coordinates / resolution;
	float3 c = (uv.x>(sin(k)*0.5f+0.5f) ) ? make_float3(1.0f,0.0f,0.0f) : make_float3(0.0f, 0.0f, 1.0f);
	float colour;
	unsigned char bytes[] = {(unsigned char)(c.x*255+0.5),(unsigned char)(c.y*255+0.5),(unsigned char)(c.z*255+0.5),1};
	memcpy(&colour, &bytes, sizeof(colour));
	output[i] = make_float3(x, y, colour);
}

int main()
{
	ShowCursor(0);
	unsigned int size = width * height * sizeof(float3);
	cudaMalloc(&device, size);
	dim3 block(8, 8, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	PIXELFORMATDESCRIPTOR pfd = { 0, 0, PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP | WS_VISIBLE | WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	glOrtho(0.0, width, 0.0, height, -1.0, 1.0);
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &buffer);
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, buffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, size, 0, 0x88EA);
	cudaGLRegisterBufferObject(buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	float s = GetTickCount();
	do
	{
		timer = (GetTickCount()-s)*0.001f;
		cudaGLMapBufferObject((void**)&device, buffer);
		rendering <<< grid, block >>>(device, timer); 
		cudaGLUnmapBufferObject(buffer);
		glDrawArrays(GL_POINTS, 0, width * height);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	cudaFree(device);
	return 0;
}