// Author: Przemyslaw Zaworski
// nvcc -x cu -o plasma.exe plasma.cl -luser32 -lgdi32 -lOpenCL

#include <windows.h>
#include <CL/cl.h>

#define width 1280
#define height 720

static const char* ComputeKernel = 

"float3 lerp(float3 a, float3 b, float w)"
"{"
	"return (float3)(mad(w,b-a,a));"
"}"

"float3 hash(float2 p)"
"{"
	"float3 d = (float3) (p.x * .1031f, p.y * .1030f, p.x * .0973f);"
	"float3 p3 = (float3) (d - floor(d));"
	"float k = p3.x * (p3.y+19.19f) + p3.y * (p3.x+19.19f) + p3.z * (p3.z+19.19f);"
	"p3 = (float3)(p3.x+k, p3.y+k, p3.z+k);"
	"float3 q = (float3)((p3.x + p3.y) * p3.z, (p3.x + p3.z) * p3.y, (p3.y + p3.z) * p3.x);"
	"return (float3) (q - floor(q));"
"}"

"float3 noise(float2 p)"
"{"
	"float2 ip = floor(p);"
	"float2 u = (float2)(p - floor(p));"
	"u = (float2)(u.x*u.x*(3.0f-2.0f*u.x), u.y*u.y*(3.0f-2.0f*u.y));"
	"float3 res = lerp(lerp(hash(ip),hash((float2)(ip.x+1.0f,ip.y)),u.x), lerp(hash((float2)(ip.x,ip.y+1.0f)),hash((float2)(ip.x+1.0f,ip.y+1.0f)),u.x),u.y);"
	"return (float3)(res * res);"
"}"

"float3 fbm(float2 p) "
"{"
	"float3 v = (float3)(0.0f, 0.0f, 0.0f);"
	"float3 a = (float3)(0.5f, 0.5f, 0.5f);"
	"for (int i = 0; i < 4; ++i)" 
	"{"
		"v = (float3)(v + a * noise(p));"
		"p = (float2)((0.87f * p.x -0.48f * p.y) * 2.0f, (0.48f * p.x + 0.87f * p.y) * 2.0f);"
		"a = (float3)(a * (float3)(0.5f, 0.5f, 0.5f));"
	"}"
	"return v;"
"}"

"float3 pattern (float2 p, float iTime)"
"{"
	"float3 q = fbm((float2)(p.x+5.2f, p.y+1.3f));"
	"float3 r = fbm((float2)(p.x+4.0f*q.x -iTime*0.5f, p.y+4.0f*q.y+iTime*0.3f));"
	"return fbm((float2)(p.x+8.0f*r.x, p.y+8.0f*r.z));"
"}"

"float2 gradient(float2 uv, float delta, float iTime)"
"{"   
	"float3 a = pattern((float2)(uv.x + delta, uv.y), iTime);"
	"float3 b = pattern((float2)(uv.x - delta, uv.y), iTime);"
	"float3 c = pattern((float2)(uv.x, uv.y + delta), iTime);"
	"float3 d = pattern((float2)(uv.x, uv.y - delta), iTime);"
	"return (float2)(length(a)-length(b), length(c)-length(d))/delta;"
"}"

"__kernel void mainImage(__global uchar4 *fragColor, float iTime)"
"{"
	"unsigned int id = get_global_id(0);"
	"int2 iResolution = (int2)(1280, 720);"
	"int2 fragCoord = (int2)(id % iResolution.x, id / iResolution.x);" 
	"float2 uv = (float2)(fragCoord.x / (float)iResolution.x * 3.0f, fragCoord.y / (float)iResolution.y * 3.0f);"
	"float3 n = normalize((float3)(gradient(uv,1.0f/(float)iResolution.y,iTime),100.0f));"
	"float3 l = normalize((float3)(1.0f,1.0f,2.0f));"
	"float s = pow(clamp(-(l.z - 2.0f * n.z * dot(n,l)), 0.0f, 1.0f), 36.0f) * 2.5f;"
	"float3 q = (float3)(clamp(pattern(uv,iTime) + (float3)(s, s, s), (float3)(0.0f,0.0f,0.0f), (float3)(1.0f,1.0f,1.0f)));"
	"fragColor[id] = (uchar4)(q.z * 255, q.y * 255, q.x * 255, 255);"
"}";

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if( uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE) )
	{
		PostQuitMessage(0); return 0;
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
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "OpenCL Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
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
		clEnqueueReadBuffer (queue, buffer, 0, 0, width*height*sizeof(uchar4), host, 0, 0, 0);		
		StretchDIBits(hdc,0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	return 0;
}