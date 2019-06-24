// Author: Przemyslaw Zaworski
// nvcc -x cu -o waves.exe waves.cl -luser32 -lgdi32 -lOpenCL
// ToDo: Correct order in RGBA components. To test, check: "fragColor[id] = (uchar4)(255, 0, 0, 255);" (should be red, not blue)

#include <windows.h>
#include <CL/cl.h>

#define width 1280
#define height 720

static const char* ComputeKernel = 
"__kernel void mainImage(__global uchar4 *fragColor, float iTime)"
"{"
	"unsigned int id = get_global_id(0);"
	"int2 iResolution = (int2)(1280, 720);"
	"int2 fragCoord = (int2)(id % iResolution.x, id / iResolution.x);" 
	"float2 uv = (float2)(fragCoord.x / (float)iResolution.x * 8.0f, fragCoord.y / (float)iResolution.y * 8.0f);"
	"for(int j=1; j<4; j++)"
	"{" 
		"uv.x += sin(iTime + uv.y * (float)j);"
		"uv.y += cos(iTime + uv.x * (float)j);"
	"}"
	"float3 q = (float3)(0.5+0.5*cos(4.0+uv.x), 0.5+0.5*cos(4.0+uv.y+2.0), 0.5+0.5*cos(4.0+uv.x+4.0));"
	"fragColor[id] = (uchar4)(q.x * 255, q.y * 255, q.z * 255, 255);"
"}";

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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "OpenCL Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "OpenCL Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0);
	const BITMAPINFO bmi = { {sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0} };
	unsigned char* host = (unsigned char*) malloc(width*height*sizeof(uchar4));
	size_t bytes;	
	cl_uint platformCount;
	clGetPlatformIDs(0, 0, &platformCount);
	cl_platform_id* platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * platformCount);
	clGetPlatformIDs(platformCount, platforms, 0);
	cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};
	cl_context context = clCreateContextFromType (properties, CL_DEVICE_TYPE_GPU, 0, 0, 0);
	clGetContextInfo (context, CL_CONTEXT_DEVICES, 0, 0, &bytes);
	cl_device_id* info = (cl_device_id*) malloc (bytes);	
	clGetContextInfo (context, CL_CONTEXT_DEVICES, bytes, info, 0);	
	cl_program program = clCreateProgramWithSource (context, 1, &ComputeKernel, 0, 0);
	clBuildProgram (program, 0, 0, 0, 0, 0);
	cl_kernel kernel = clCreateKernel (program, "mainImage", 0);	
	cl_command_queue queue = clCreateCommandQueue (context, info[0], 0, 0); 
	cl_mem buffer = clCreateBuffer (context, CL_MEM_WRITE_ONLY, width*height*sizeof(uchar4), 0, 0);
	clSetKernelArg (kernel, 0, sizeof(cl_mem), (void*)&buffer);
	size_t size = width * height;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float time = GetTickCount()*0.001f;
		clSetKernelArg (kernel, 1, sizeof(cl_float), (void*)&time);
		clEnqueueNDRangeKernel(queue,kernel, 1, 0, &size, 0, 0, 0, 0);	
		clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0, width*height*sizeof(uchar4), host, 0, 0, 0);		
		StretchDIBits(GetDC(hwnd),0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	return 0;
}