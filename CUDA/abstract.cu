// nvcc -o abstract.exe abstract.cu -arch=sm_30 user32.lib gdi32.lib
#include <windows.h>

#define width 1280
#define height 720

static const BITMAPINFO bmi = { {sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0} };
static unsigned char host[width*height*4];

__global__ void mainImage(uchar4 *fragColor, float iTime)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;
	float2 iResolution = make_float2((float)width, (float)height);
	float2 fragCoord = make_float2((float)x, (float)y);
	float2 uv = make_float2(fragCoord.x/iResolution.x*8.0f, fragCoord.y/iResolution.y*8.0f);
	for(int j=1; j<4; j++)
	{ 
		uv.x += sin(iTime + uv.y * float(j));
		uv.y += cos(iTime + uv.x * float(j));
	}
	float3 q = make_float3(0.5+0.5*cos(4.0+uv.x), 0.5+0.5*cos(4.0+uv.y+2.0), 0.5+0.5*cos(4.0+uv.x+4.0));
	fragColor[i] = make_uchar4(q.x*255, q.y*255, q.z*255, 1);
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "CUDA Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "CUDA Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0);
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
		mainImage<<<grid, block>>>(device, GetTickCount()*0.001f);
		cudaThreadSynchronize();
		cudaMemcpy(host, device, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		StretchDIBits(GetDC(hwnd),0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	cudaFree( device );
	return 0;
}