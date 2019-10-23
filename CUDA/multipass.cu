// nvcc -o multipass.exe multipass.cu -arch=sm_30 user32.lib gdi32.lib
#include <windows.h>

#define width 1920
#define height 1080

__global__ void mainImage(uchar4 *fragColor, float iTime)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;
	float2 iResolution = make_float2(width, height);
	float2 fragCoord = make_float2(x, y);
	float2 uv = make_float2((2.0f*fragCoord.x-iResolution.x)/iResolution.y, (2.0f*fragCoord.y-iResolution.y)/iResolution.y);
	float t = floorf(iTime+10.0f); 
	float hx = (sinf(t*127.1f+t*311.7f)*43758.5453f) - floorf(sinf(t*127.1f+t*311.7f)*43758.5453f);
	float hy = (sinf(t*269.5f+t*183.3f)*43758.5453f) - floorf(sinf(t*269.5f+t*183.3f)*43758.5453f);
	float s = 1.0f-((sqrtf((uv.x-(hx*2.0f-1.0f))*(uv.x-(hx*2.0f-1.0f))+(uv.y-(hy*2.0f-1.0f))*(uv.y-(hy*2.0f-1.0f)))-0.07f) >= 0.0f);
	float buffer = fragColor[i].x;
	fragColor[i] = make_uchar4(fminf(s*255.0f + buffer, 255.0f), 0, 0, 255);
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg == WM_KEYUP && wParam == VK_ESCAPE)
	{
		PostQuitMessage(0); return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	ShowCursor(0);
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "CUDA Demo"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "CUDA Demo", WS_VISIBLE|WS_POPUP, 0, 0, width, height, 0, 0, 0, 0));
	uchar4 *device; 
	cudaMalloc( (void**)&device, width*height*4 );
	dim3 block(8, 8);
	dim3 grid(width/8, height/8);
	BITMAPINFO bmi = {{sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0}};
	static unsigned char host[width*height*4];
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		mainImage<<<grid, block>>>(device, GetTickCount()*0.001f);
		cudaMemcpy(host, device, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		StretchDIBits(hdc,0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	cudaFree( device );
	return 0;
}