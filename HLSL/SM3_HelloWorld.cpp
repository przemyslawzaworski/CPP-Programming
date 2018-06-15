//g++ -s -o SM3_HelloWorld.exe SM3_HelloWorld.cpp  "-IC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include" "-LC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Lib\x86" -ld3d9 -ld3dx9
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>

#define XRES 1920
#define YRES 1080

static const char VertexShaderCode[] = \
"float4 VSMain(float4 P:POSITION):POSITION {return P;};" ;

static const char PixelShaderCode[] = \
"float T : register(c0);"
"void PSMain(float2 U:VPOS, out float4 S:COLOR) "
"{"
	"U = U/float2(1920,1080);"
	"S = lerp(float4(0,sin(T)*0.5+0.5,0,1),0.5,step(U.x,0.5));"
"}" ;

int main()
{
	IDirect3DVertexShader9 *VertexShader;
	IDirect3DPixelShader9 *PixelShader;
	ID3DXBuffer *VSBuffer, *PSBuffer;	
	IDirect3DDevice9 *d3dDevice;	
	IDirect3D9 *d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS W = {XRES,YRES,D3DFMT_A8R8G8B8,0,D3DMULTISAMPLE_NONE,0,D3DSWAPEFFECT_DISCARD,0,0,1,D3DFMT_D24S8,0,0,D3DPRESENT_INTERVAL_IMMEDIATE};	
	W.hDeviceWindow = CreateWindow("static",0,WS_POPUP|WS_VISIBLE,0,0,XRES,YRES,0,0,0,0);
	d3d->CreateDevice(D3DADAPTER_DEFAULT,D3DDEVTYPE_HAL,W.hDeviceWindow,D3DCREATE_HARDWARE_VERTEXPROCESSING,&W,&d3dDevice);
	ShowCursor(0);  
	D3DXCompileShader(VertexShaderCode,sizeof(VertexShaderCode),0,0,"VSMain","vs_3_0",D3DXSHADER_USE_LEGACY_D3DX9_31_DLL,&VSBuffer,0,0);
	D3DXCompileShader(PixelShaderCode,sizeof(PixelShaderCode),0,0,"PSMain","ps_3_0",D3DXSHADER_USE_LEGACY_D3DX9_31_DLL,&PSBuffer,0,0);
	d3dDevice->CreateVertexShader((DWORD*)VSBuffer->GetBufferPointer(), &VertexShader);
	d3dDevice->CreatePixelShader((DWORD*)PSBuffer->GetBufferPointer(), &PixelShader);
	float quad[20] = {1,-1,0,1,0,-1,-1,0,0,0,1,1,0,1,1,-1,1,0,0,1};
	do 
	{
		d3dDevice->BeginScene();
		d3dDevice->SetVertexShader(VertexShader);
		d3dDevice->SetPixelShader(PixelShader);
		d3dDevice->SetFVF(D3DFVF_XYZ);
		float timer[1] = {GetTickCount()*0.001f};
		d3dDevice->SetPixelShaderConstantF(0, timer, 1);
		d3dDevice->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, 5*sizeof(float));
		d3dDevice->EndScene();
		d3dDevice->Present(NULL, NULL, NULL, NULL);		
	}
	while ( !GetAsyncKeyState(VK_ESCAPE) );
	ExitProcess(0);
	return 0;
}
