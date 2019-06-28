// nvcc -o blocks.exe blocks.cu -arch=sm_30 user32.lib gdi32.lib
#include <windows.h>

#define width 1280
#define height 720

__global__ void mainImage(uchar4 *fragColor, float iTime)
{	
	asm volatile(
	".reg .b16 	%rs<5>;"
	".reg .f32 	%f<93>;"
	".reg .b32 	%r<16>;"
	".reg .b64 	%rd<5>;"
	"ld.param.u64 	%rd1, [_Z9mainImageP6uchar4f_param_0];"
	"ld.param.f32 	%f1, [_Z9mainImageP6uchar4f_param_1];"
	"cvta.to.global.u64 	%rd2, %rd1;"
	"mov.u32 	%r1, %ctaid.x;"
	"mov.u32 	%r2, %ntid.x;"
	"mov.u32 	%r3, %tid.x;"
	"mad.lo.s32 	%r4, %r2, %r1, %r3;"
	"mov.u32 	%r5, %ctaid.y;"
	"mov.u32 	%r6, %ntid.y;"
	"mov.u32 	%r7, %tid.y;"
	"mad.lo.s32 	%r8, %r6, %r5, %r7;"
	"mov.u32 	%r9, 720;"
	"sub.s32 	%r10, %r9, %r8;"
	"mad.lo.s32 	%r11, %r10, 1280, %r4;"
	"add.s32 	%r12, %r11, -1280;"
	"cvt.rn.f32.u32	%f2, %r4;"
	"div.rn.f32 	%f3, %f2, 0f44A00000;"
	"cvt.rn.f32.u32	%f4, %r8;"
	"div.rn.f32 	%f5, %f4, 0f44340000;"
	"fma.rn.f32 	%f6, %f5, 0f40000000, %f1;"
	"add.f32 	%f7, %f6, 0fBFC90FDA;"
	"div.rn.f32 	%f8, %f7, 0f40C90FDA;"
	"cvt.rmi.f32.f32	%f9, %f8;"
	"sub.f32 	%f10, %f8, %f9;"
	"fma.rn.f32 	%f11, %f10, 0f40000000, 0fBF800000;"
	"abs.f32 	%f12, %f11;"
	"mul.f32 	%f13, %f12, %f12;"
	"add.f32 	%f14, %f12, %f12;"
	"mov.f32 	%f15, 0f40400000;"
	"sub.f32 	%f16, %f15, %f14;"
	"mul.f32 	%f17, %f13, %f16;"
	"fma.rn.f32 	%f18, %f17, 0f40000000, 0fBF800000;"
	"div.rn.f32 	%f19, %f18, 0f40A00000;"
	"add.f32 	%f20, %f3, %f19;"
	"fma.rn.f32 	%f21, %f20, 0f40000000, %f1;"
	"add.f32 	%f22, %f21, 0f3FC90FDB;"
	"add.f32 	%f23, %f22, 0fBFC90FDA;"
	"div.rn.f32 	%f24, %f23, 0f40C90FDA;"
	"cvt.rmi.f32.f32	%f25, %f24;"
	"sub.f32 	%f26, %f24, %f25;"
	"fma.rn.f32 	%f27, %f26, 0f40000000, 0fBF800000;"
	"abs.f32 	%f28, %f27;"
	"mul.f32 	%f29, %f28, %f28;"
	"add.f32 	%f30, %f28, %f28;"
	"sub.f32 	%f31, %f15, %f30;"
	"mul.f32 	%f32, %f29, %f31;"
	"fma.rn.f32 	%f33, %f32, 0f40000000, 0fBF800000;"
	"div.rn.f32 	%f34, %f33, 0f40A00000;"
	"add.f32 	%f35, %f5, %f34;"
	"mul.f32 	%f36, %f20, 0f41800000;"
	"cvt.rmi.f32.f32	%f37, %f36;"
	"mul.f32 	%f38, %f37, 0f3D800000;"
	"mul.f32 	%f39, %f35, 0f41800000;"
	"cvt.rmi.f32.f32	%f40, %f39;"
	"mul.f32 	%f41, %f40, 0f3D800000;"
	"mul.f32 	%f42, %f41, 0f439BD99A;"
	"fma.rn.f32 	%f43, %f38, 0f42FE3333, %f42;"
	"mul.f32 	%f44, %f41, 0f43374CCD;"
	"fma.rn.f32 	%f45, %f38, 0f4386C000, %f44;"
	"mul.f32 	%f46, %f41, 0f43B9F333;"
	"fma.rn.f32 	%f47, %f38, 0f43D1999A, %f46;"
	"add.f32 	%f48, %f43, 0fBFC90FDA;"
	"div.rn.f32 	%f49, %f48, 0f40C90FDA;"
	"cvt.rmi.f32.f32	%f50, %f49;"
	"sub.f32 	%f51, %f49, %f50;"
	"fma.rn.f32 	%f52, %f51, 0f40000000, 0fBF800000;"
	"abs.f32 	%f53, %f52;"
	"mul.f32 	%f54, %f53, %f53;"
	"add.f32 	%f55, %f53, %f53;"
	"sub.f32 	%f56, %f15, %f55;"
	"mul.f32 	%f57, %f54, %f56;"
	"fma.rn.f32 	%f58, %f57, 0f40000000, 0fBF800000;"
	"mul.f32 	%f59, %f58, 0f472AEE8C;"
	"add.f32 	%f60, %f45, 0fBFC90FDA;"
	"div.rn.f32 	%f61, %f60, 0f40C90FDA;"
	"cvt.rmi.f32.f32	%f62, %f61;"
	"sub.f32 	%f63, %f61, %f62;"
	"fma.rn.f32 	%f64, %f63, 0f40000000, 0fBF800000;"
	"abs.f32 	%f65, %f64;"
	"mul.f32 	%f66, %f65, %f65;"
	"add.f32 	%f67, %f65, %f65;"
	"sub.f32 	%f68, %f15, %f67;"
	"mul.f32 	%f69, %f66, %f68;"
	"fma.rn.f32 	%f70, %f69, 0f40000000, 0fBF800000;"
	"mul.f32 	%f71, %f70, 0f472AEE8C;"
	"add.f32 	%f72, %f47, 0fBFC90FDA;"
	"div.rn.f32 	%f73, %f72, 0f40C90FDA;"
	"cvt.rmi.f32.f32	%f74, %f73;"
	"sub.f32 	%f75, %f73, %f74;"
	"fma.rn.f32 	%f76, %f75, 0f40000000, 0fBF800000;"
	"abs.f32 	%f77, %f76;"
	"mul.f32 	%f78, %f77, %f77;"
	"add.f32 	%f79, %f77, %f77;"
	"sub.f32 	%f80, %f15, %f79;"
	"mul.f32 	%f81, %f78, %f80;"
	"fma.rn.f32 	%f82, %f81, 0f40000000, 0fBF800000;"
	"mul.f32 	%f83, %f82, 0f472AEE8C;"
	"cvt.rmi.f32.f32	%f84, %f59;"
	"sub.f32 	%f85, %f59, %f84;"
	"mul.f32 	%f86, %f85, 0f437F0000;"
	"cvt.rzi.u32.f32	%r13, %f86;"
	"cvt.rmi.f32.f32	%f87, %f71;"
	"sub.f32 	%f88, %f71, %f87;"
	"mul.f32 	%f89, %f88, 0f437F0000;"
	"cvt.rzi.u32.f32	%r14, %f89;"
	"cvt.rmi.f32.f32	%f90, %f83;"
	"sub.f32 	%f91, %f83, %f90;"
	"mul.f32 	%f92, %f91, 0f437F0000;"
	"cvt.rzi.u32.f32	%r15, %f92;"
	"mul.wide.u32 	%rd3, %r12, 4;"
	"add.s64 	%rd4, %rd2, %rd3;"
	"cvt.u16.u32	%rs1, %r15;"
	"cvt.u16.u32	%rs2, %r14;"
	"cvt.u16.u32	%rs3, %r13;"
	"mov.u16 	%rs4, 255;"
	"st.global.v4.u8 [%rd4], {%rs3, %rs2, %rs1, %rs4};"
	"ret;"
	);
}

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE))
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "CUDA PTX Demo"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "CUDA PTX Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0);
	const BITMAPINFO bmi = { {sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0} };	
	unsigned char* host = (unsigned char*) malloc(width*height*sizeof(uchar4));
	uchar4 *device;
	cudaMalloc( (void**)&device, width*height*sizeof(uchar4) );
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
		cudaDeviceSynchronize();
		cudaMemcpy(host, device, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost);
		StretchDIBits(GetDC(hwnd),0,0,width,height,0,0,width,height,host,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	cudaFree( device );
	return 0;
}