/*  
Author: Przemyslaw Zaworski
OpenCL PTX assembly demo
Compile: nvcc -x cu -o dawn.exe dawn.cl -luser32 -lgdi32 -lOpenCL
Optimizations:

float Sinus (float x)
{
	float ptr = 0.0f;
	float x = fabs(fract((x - 0.017453292*90.0) / (0.017453292*360.0),&ptr)*2.0-1.0);     
	return x * x * (3.0 - 2.0 * x) * 2.0 - 1.0;
}

Cosinus (x) = Sinus(x+1.57079632679);

normalize(N) = rsqrt(N.x*N.x+N.y*N.y+N.z*N.z) * N;
*/

#include <windows.h>
#include <CL/cl.h>

#define width 1280
#define height 720

static const char* ComputeKernel =
"__kernel void mainImage(__global uchar4 *fragColor, float iTime)"
"{"	
	"asm(\""
	".reg .pred 	%p<3>;"
	".reg .b16 	%rs<3>;"
	".reg .f32 	%f<49>;"
	".reg .b32 	%r<10>;"
	".reg .f64 	%fd<38>;"
	".reg .b64 	%rd<6>;"
	"ld.param.u64 	%rd1, [mainImage_param_0];"
	"ld.param.f32 	%f1, [mainImage_param_1];"
	"mov.b32	%r1, %envreg3;"
	"mov.u32 	%r2, %ntid.x;"
	"mov.u32 	%r3, %ctaid.x;"
	"mad.lo.s32 	%r4, %r3, %r2, %r1;"
	"mov.u32 	%r5, %tid.x;"
	"add.s32 	%r6, %r4, %r5;"
	"mul.wide.u32 	%rd2, %r6, -858993459;"
	"shr.u64 	%rd3, %rd2, 42;"
	"cvt.u32.u64	%r7, %rd3;"
	"mul.lo.s32 	%r8, %r7, 1280;"
	"sub.s32 	%r9, %r6, %r8;"
	"cvt.rn.f64.s32	%fd1, %r9;"
	"fma.rn.f64 	%fd2, %fd1, 0d4000000000000000, 0dC094000000000000;"
	"add.f64 	%fd3, %fd2, 0dBFF0000000000000;"
	"div.rn.f64 	%fd4, %fd3, 0d4086800000000000;"
	"cvt.rn.f64.s32	%fd5, %r7;"
	"fma.rn.f64 	%fd6, %fd5, 0d4000000000000000, 0dC086800000000000;"
	"div.rn.f64 	%fd7, %fd6, 0d4086800000000000;"
	"cvt.f64.f32	%fd8, %f1;"
	"fma.rn.f64 	%fd9, %fd8, 0d3FC999999999999A, 0d3FF921FB5443D6F4;"
	"cvt.rn.f32.f64	%f2, %fd9;"
	"cvt.f64.f32	%fd10, %f2;"
	"sub.f64 	%fd11, %fd10, 0d3FF921FB47B47491;"
	"div.rn.f64 	%fd12, %fd11, 0d401921FB47B47491;"
	"cvt.rn.f32.f64	%f3, %fd12;"
	"cvt.rmi.f32.f32	%f4, %f3;"
	"sub.f32 	%f5, %f3, %f4;"
	"mov.f32 	%f6, 0f00000000;"
	"max.f32 	%f7, %f5, %f6;"
	"mov.f32 	%f8, 0f3F7FFFFF;"
	"min.f32 	%f9, %f7, %f8;"
	"abs.f32 	%f10, %f3;"
	"setp.gtu.f32	%p1, %f10, 0f7F800000;"
	"selp.f32	%f11, %f3, %f9, %p1;"
	"cvt.f64.f32	%fd13, %f11;"
	"fma.rn.f64 	%fd14, %fd13, 0d4000000000000000, 0dBFF0000000000000;"
	"abs.f64 	%fd15, %fd14;"
	"cvt.rn.f32.f64	%f12, %fd15;"
	"mul.f32 	%f13, %f12, %f12;"
	"cvt.f64.f32	%fd16, %f13;"
	"cvt.f64.f32	%fd17, %f12;"
	"fma.rn.f64 	%fd18, %fd17, 0dC000000000000000, 0d4008000000000000;"
	"mul.f64 	%fd19, %fd18, %fd16;"
	"fma.rn.f64 	%fd20, %fd19, 0d4000000000000000, 0dBFF0000000000000;"
	"cvt.rn.f32.f64	%f14, %fd20;"
	"mul.f64 	%fd21, %fd8, 0d3FC999999999999A;"
	"cvt.rn.f32.f64	%f15, %fd21;"
	"cvt.f64.f32	%fd22, %f15;"
	"sub.f64 	%fd23, %fd22, 0d3FF921FB47B47491;"
	"div.rn.f64 	%fd24, %fd23, 0d401921FB47B47491;"
	"cvt.rn.f32.f64	%f16, %fd24;"
	"cvt.rmi.f32.f32	%f17, %f16;"
	"sub.f32 	%f18, %f16, %f17;"
	"max.f32 	%f19, %f18, %f6;"
	"min.f32 	%f20, %f19, %f8;"
	"abs.f32 	%f21, %f16;"
	"setp.gtu.f32	%p2, %f21, 0f7F800000;"
	"selp.f32	%f22, %f16, %f20, %p2;"
	"cvt.f64.f32	%fd25, %f22;"
	"fma.rn.f64 	%fd26, %fd25, 0d4000000000000000, 0dBFF0000000000000;"
	"abs.f64 	%fd27, %fd26;"
	"cvt.rn.f32.f64	%f23, %fd27;"
	"mul.f32 	%f24, %f23, %f23;"
	"cvt.f64.f32	%fd28, %f24;"
	"cvt.f64.f32	%fd29, %f23;"
	"fma.rn.f64 	%fd30, %fd29, 0dC000000000000000, 0d4008000000000000;"
	"mul.f64 	%fd31, %fd30, %fd28;"
	"fma.rn.f64 	%fd32, %fd31, 0d4000000000000000, 0dBFF0000000000000;"
	"cvt.rn.f32.f64	%f25, %fd32;"
	"mul.f32 	%f26, %f25, 0f42480000;"
	"mul.f32 	%f27, %f14, 0f42480000;"
	"mul.f32 	%f28, %f27, %f27;"
	"fma.rn.f32 	%f29, %f6, 0f00000000, %f28;"
	"fma.rn.f32 	%f30, %f26, %f26, %f29;"
	"rsqrt.approx.f32 	%f31, %f30;"
	"mul.f32 	%f32, %f26, %f31;"
	"mul.f32 	%f33, %f31, 0f00000000;"
	"mul.f32 	%f34, %f27, %f31;"
	"cvt.rn.f32.f64	%f35, %fd7;"
	"mul.f32 	%f36, %f35, %f35;"
	"cvt.rn.f32.f64	%f37, %fd4;"
	"fma.rn.f32 	%f38, %f37, %f37, %f36;"
	"cvt.f64.f32	%fd33, %f38;"
	"mov.f64 	%fd34, 0d3FD0000000000000;"
	"sub.f64 	%fd35, %fd34, %fd33;"
	"abs.f64 	%fd36, %fd35;"
	"sqrt.rn.f64 	%fd37, %fd36;"
	"cvt.rn.f32.f64	%f39, %fd37;"
	"fma.rn.f32 	%f40, %f39, %f39, %f38;"
	"rsqrt.approx.f32 	%f41, %f40;"
	"mul.f32 	%f42, %f39, %f41;"
	"mul.f32 	%f43, %f37, %f41;"
	"mul.f32 	%f44, %f35, %f41;"
	"mul.f32 	%f45, %f34, %f44;"
	"fma.rn.f32 	%f46, %f43, %f33, %f45;"
	"fma.rn.f32 	%f47, %f42, %f32, %f46;"
	"mul.f32 	%f48, %f47, 0f437F0000;"
	"mul.wide.u32 	%rd4, %r6, 4;"
	"add.s64 	%rd5, %rd1, %rd4;"
	"cvt.rzi.u16.f32	%rs1, %f48;"
	"mov.u16 	%rs2, 255;"
	"st.global.v4.u8 [%rd5], {%rs1, %rs1, %rs1, %rs2};"
	"ret;"
	"\");"
"}";

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
	float s = GetTickCount()*0.001f;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float time = GetTickCount()*0.001f - s;
		clSetKernelArg (kernel, 1, sizeof(cl_float), (void*)&time);
		clEnqueueNDRangeKernel(queue,kernel, 1, 0, &size, 0, 0, 0, 0);	
		clEnqueueReadBuffer (queue, buffer, CL_TRUE, 0, width*height*sizeof(uchar4), host, 0, 0, 0);		
		StretchDIBits(GetDC(hwnd),0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	return 0;
}