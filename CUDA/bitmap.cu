// nvcc -o bitmap.exe bitmap.cu -ID:\CUDA\Samples\common\inc -arch=sm_30 user32.lib gdi32.lib
#include <windows.h>
#include <helper_math.h> 
#include <cuda_runtime.h>

#define width 1280
#define height 720

static const BITMAPINFO bmi = { {sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0} };
static unsigned char host[width*height*4];

__global__ void mainImage(uchar4 *fragColor)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;
	float2 iResolution = make_float2((float)width, (float)height);
	float2 fragCoord = make_float2((float)x, (float)y);
	float2 uv = fragCoord / iResolution;
	float2 p = floorf(uv*16.0)/16.0;
	float3 q = make_float3( dot(p,make_float2(127.1,311.7)), dot(p,make_float2(269.5,183.3)), dot(p,make_float2(419.2,371.9)) ); 
	float3 k = make_float3(sin(q.x)*43758.5453, sin(q.y)*43758.5453, sin(q.z)*43758.5453);
	fragColor[i] = make_uchar4(fracf(k.x)*255, fracf(k.y)*255, fracf(k.z)*255, 1);
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_KEYUP && wParam == VK_ESCAPE)
	{
		PostQuitMessage(0);
		return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASSEX WndClassEx = { 0 };
	WndClassEx.cbSize = sizeof(WndClassEx);
	WndClassEx.lpfnWndProc = WindowProc;
	WndClassEx.lpszClassName = "CUDA Demo";
	RegisterClassEx(&WndClassEx);
	HWND hwnd = CreateWindowEx(WS_EX_LEFT, WndClassEx.lpszClassName, NULL, WS_POPUP, 0, 0, width, height, 0, 0, NULL, NULL);
	HDC hdc = GetDC(hwnd);
	ShowWindow(hwnd, SW_SHOW);
	uchar4 *device; 
	cudaMalloc( (void**)&device, width*height*4 );
	dim3 block(8, 8);
	dim3 grid(width/8, height/8);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		mainImage<<<grid, block>>>(device);
		cudaThreadSynchronize();
		cudaMemcpy(host, device, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		StretchDIBits(hdc,0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	cudaFree( device );
	return 0;
}