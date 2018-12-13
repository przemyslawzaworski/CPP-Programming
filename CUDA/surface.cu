// nvcc -o surface.exe surface.cu -ID:\CUDA\Samples\common\inc -lopengl32 -arch=sm_30 user32.lib gdi32.lib
// Przemyslaw Zaworski 13.12.2018

#include <windows.h>
#include <cuda_gl_interop.h>
#include <helper_math.h>
#include <GL/gl.h>

#define width 1280
#define height 720
typedef int (WINAPI* PFNWGLSWAPINTERVALEXTPROC)(int i);
surface<void, cudaSurfaceType2D> RenderSurface;

__global__ void mainImage (float k)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float2 iResolution = make_float2((float)width, (float)height);
	float2 fragCoord = make_float2((float)x, (float)y);
	float2 uv = fragCoord / iResolution;
	uchar4 fragColor = (uv.x>(sin(k)*0.5f+0.5f)) ? make_uchar4(0,0,255,255) : make_uchar4(255,0,0,255);	
	surf2Dwrite(fragColor, RenderSurface, x*sizeof(uchar4), y, cudaBoundaryModeClamp);
}

int main()
{ 
	dim3 block(8, 8, 1);
	dim3 grid(width/block.x, height/block.y, 1);
	cudaGraphicsResource *resource;
	cudaArray *image;
	PIXELFORMATDESCRIPTOR pfd = {0,0,PFD_DOUBLEBUFFER};
	HDC hdc = GetDC(CreateWindow("static", 0, WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress ("wglSwapIntervalEXT")) (0);	
	glOrtho(0, width, 0, height, -1, 1);
	unsigned int store;
	glGenTextures(1, &store);
	glBindTexture(GL_TEXTURE_2D, store);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glEnable(GL_TEXTURE_2D);
	cudaGraphicsGLRegisterImage(&resource, store, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsSubResourceGetMappedArray(&image, resource, 0, 0);
	cudaBindSurfaceToArray(RenderSurface, image);	
	do
	{
		mainImage <<< grid, block >>> (GetTickCount()*0.001f);
		cudaDeviceSynchronize();
		glBegin(GL_QUADS);
		glTexCoord2i(0, 0); glVertex2i(0, 0);
		glTexCoord2i(0, 1); glVertex2i(0, height);
		glTexCoord2i(1, 1); glVertex2i(width, height);
		glTexCoord2i(1, 0); glVertex2i(width, 0);
		glEnd();
		wglSwapLayerBuffers(hdc,WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	cudaGraphicsUnmapResources(1, &resource, 0);
	cudaGraphicsUnregisterResource(resource);
	return 0;
}