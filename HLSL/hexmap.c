// cl.exe hexmap.c /I"C:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include" /link /LIBPATH:"C:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Lib\x64" d3d9.lib d3dx9.lib user32.lib kernel32.lib gdi32.lib
#include <d3dx9.h>

static const char PixelShader[] = \
	"ps_3_0 \n"
	"def c1, 0.00925925933, 1.15470064, 0.577350318, 0.333333343 \n"
	"def c2, -0.5, 0.5, 4.47034836e-007, 0 \n"
	"def c3, 0.159154937, 0.25, 6.28318548, -3.14159274 \n"
	"def c4, -2.52398507e-007, 2.47609005e-005, -0.00138883968, 0.0416666418 \n"
	"def c5, 3, -1, -2, -0 \n"
	"dcl vPos.xy \n"
	"mov r0.x, c1.x \n"
	"mad r0.xy, vPos, r0.x, c0.x \n"
	"mul r1.x, r0.x, c1.y \n"
	"mad r1.z, r0.x, c1.z, r0.y \n"
	"frc r0.xy, r1.zxzw \n"
	"add r0.zw, -r0.xyyx, r1.xyxz \n"
	"add r0.xy, -r0.yxzw, r0 \n"
	"cmp r0.xy, r0, c5.y, c5.w \n"
	"add r1.x, r0.w, r0.z \n"
	"mul r1.y, r1.x, c1.w \n"
	"frc r1.z, r1.y \n"
	"add r1.y, -r1.z, r1.y \n"
	"mad r1.x, r1.y, -c5.x, r1.x \n"
	"add r1.xy, r1.x, c5.yzzw \n"
	"cmp r0.xy, r1.y, r0, -c5.w \n"
	"cmp r1.x, r1.x, -c5.y, -c5.w \n"
	"add r0.zw, r0, r1.x \n"
	"add r0.xy, r0, r0.zwzw \n"
	"mad r0.xy, r0, c3.x, c3.y \n"
	"frc r0.xy, r0 \n"
	"mad r0.xy, r0, c3.z, c3.w \n"
	"mul r0.xy, r0, r0 \n"
	"mad r0.zw, r0.xyxy, c4.x, c4.y \n"
	"mad r0.zw, r0.xyxy, r0, c4.z \n"
	"mad r0.zw, r0.xyxy, r0, c4.w \n"
	"mad r0.zw, r0.xyxy, r0, c2.x \n"
	"mad r0.xy, r0, r0.zwzw, -c5.y \n"
	"mul r0.xy, r0, c2.y \n"
	"mov r0.zw, c2.z \n"
	"add oC0, r0, c2.y \n";

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

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	ShowCursor(0);
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, " "};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, " ", WS_VISIBLE|WS_POPUP, 0, 0, 1920, 1080, 0, 0, 0, 0);
	LPDIRECT3DPIXELSHADER9 ps;
	LPD3DXBUFFER buffer;
	LPDIRECT3DDEVICE9 device;
	LPDIRECT3D9 d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS d3p = {1920, 1080, D3DFMT_A8R8G8B8, 1, 0, 0, 1, hwnd, 0, 1, D3DFMT_D24S8, 0, 0, 0x80000000L};
	d3d->lpVtbl->CreateDevice(d3d, 0, D3DDEVTYPE_HAL, hwnd, 0x00000040L, &d3p, &device);  
	D3DXAssembleShader(PixelShader, sizeof(PixelShader), 0, 0, (1 << 16), &buffer, 0);
	device->lpVtbl->CreatePixelShader(device, (DWORD*)buffer->lpVtbl->GetBufferPointer(buffer), &ps);
	device->lpVtbl->SetPixelShader(device, ps);
	device->lpVtbl->SetFVF(device, D3DFVF_XYZ);
	float quad[20] = {1,-1,0,1,0,-1,-1,0,0,0,1,1,0,1,1,-1,1,0,0,1}, S = GetTickCount()*0.001f;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message == WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		device->lpVtbl->BeginScene(device);
		float timer[1] = {GetTickCount()*0.001f-S};
		device->lpVtbl->SetPixelShaderConstantF(device,0, timer, 1);
		device->lpVtbl->DrawPrimitiveUP(device, D3DPT_TRIANGLESTRIP, 2, quad, 20);
		device->lpVtbl->EndScene(device);
		device->lpVtbl->Present(device, 0, 0, 0, 0);
	}
	return 0;
}