// nvcc -o apollonian.exe apollonian.cu -arch=sm_30 user32.lib gdi32.lib
#include <windows.h>

#define width 1920
#define height 1080

__device__ float clamp (float x)
{
	return fmaxf(0.0f, fminf(255.0f, x));
}

__device__ float3 mod (float3 s, float3 m)
{
	float a = s.x - m.x * floorf(s.x/m.x);
	float b = s.y - m.y * floorf(s.y/m.y);
	float c = s.z - m.z * floorf(s.z/m.z);
	return make_float3(a, b, c);
}

__device__ float fractal (float3 p, float iTime)
{
	float i = 0.0f, s = 1.0f, k = 0.0f;
	for(p.y += iTime * 0.3f; i++ < 7.0f; s *= k)
	{
		p = make_float3(p.x - 1.0f, p.y - 1.0f, p.z - 1.0f);
		p = mod(p, make_float3(2.0f,2.0f,2.0f));
		p = make_float3(p.x - 1.0f, p.y - 1.0f, p.z - 1.0f);
		k = 1.5f / (p.x*p.x + p.y*p.y + p.z*p.z);
		p = make_float3(p.x*k, p.y*k, p.z*k);
	}
	return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z)/s - 0.01f;
}

__global__ void mainImage (uchar4 *fragColor, float iTime)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;
	float2 iResolution = make_float2(width, height);
	float2 fragCoord = make_float2(x, y);
	float3 d = make_float3((2.0f*fragCoord.x-iResolution.x)/iResolution.y, (2.0f*fragCoord.y-iResolution.y)/iResolution.y, 1.0f);
	d = make_float3(d.x / 6.0f, d.y / 6.0f, d.z / 6.0f);
	float3 o = make_float3(1.0f, 1.0f, 1.0f);
	float4 c = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int i=0; i<100; i++)
	{
		float f = fractal(o, iTime);
		o = make_float3(o.x+f*d.x, o.y+f*d.y, o.z+f*d.z);    
	}
	float m = fractal(make_float3(o.x-d.x,o.y-d.y,o.z-d.z), iTime) / powf(o.z,1.5f) * 2.5f;
	c = make_float4(c.x + m, c.y + m, c.z + m, 255.0f);
	fragColor[i] = make_uchar4( clamp(c.z * 255.0f), clamp(c.y * 255.0f), clamp(c.x * 255.0f), 255.0f);
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