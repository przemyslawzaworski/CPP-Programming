// g++ -s -o Asm.exe Asm.cpp  "-IC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include" "-LC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Lib\x86" -ld3d9 -ld3dx9
// fxc /T ps_2_0 /Fc C:\pixel.asm C:\pixel.hlsl
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#define DLL D3DXSHADER_USE_LEGACY_D3DX9_31_DLL
#define MSAA D3DMULTISAMPLE_NONE
#define VSYNC D3DPRESENT_INTERVAL_IMMEDIATE
#define SWAP D3DSWAPEFFECT_DISCARD
#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <stdio.h>

static const char VertexShaderCode[] = \
	"vs.2.0\n"
	"dcl_position v0\n"
	"mov oPos, v0\n" ;

static const char PixelShaderCode[] = \
	"ps.2.0\n"
	"def c0, 0, 0, 1, 1\n"
	"mov oC0, c0\n";

int main()
{
	LPDIRECT3DVERTEXSHADER9 VertexShader;
	LPDIRECT3DPIXELSHADER9 PixelShader;
	LPD3DXBUFFER VSBuffer, PSBuffer, PSDebug;	
	LPDIRECT3DDEVICE9 d3dDevice;	
	LPDIRECT3D9 d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS W = {1920,1080,D3DFMT_A8R8G8B8,1,MSAA,0,SWAP,0,0,1,D3DFMT_D24S8,0,0,VSYNC};	
	W.hDeviceWindow = CreateWindow("static",0,WS_POPUP|WS_VISIBLE,0,0,1920,1080,0,0,0,0);
	d3d->CreateDevice(0,D3DDEVTYPE_HAL,W.hDeviceWindow,D3DCREATE_HARDWARE_VERTEXPROCESSING,&W,&d3dDevice);
	ShowCursor(0);  
	D3DXAssembleShader(VertexShaderCode,sizeof(VertexShaderCode), 0,0, DLL, &VSBuffer, 0);
	D3DXAssembleShader(PixelShaderCode,sizeof(PixelShaderCode), 0,0, DLL, &PSBuffer, &PSDebug);
	if (PSDebug)
	{
		FILE *file;
		file = fopen ("debug.log","a");
		char *p = (char*)PSDebug->GetBufferPointer();
		fprintf (file,"\n%s error : %s \n", p );
		fclose (file);
		return 0;
	}
	d3dDevice->CreateVertexShader((DWORD*)VSBuffer->GetBufferPointer(), &VertexShader);
	d3dDevice->CreatePixelShader((DWORD*)PSBuffer->GetBufferPointer(), &PixelShader);
	d3dDevice->SetVertexShader(VertexShader);
	d3dDevice->SetPixelShader(PixelShader);
	d3dDevice->SetFVF(D3DFVF_XYZ);
	float quad[20] = {1,-1,0,1,0,-1,-1,0,0,0,1,1,0,1,1,-1,1,0,0,1};	
	do 
	{
		d3dDevice->BeginScene();
		d3dDevice->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, quad, 5*sizeof(float));
		d3dDevice->EndScene();
		d3dDevice->Present(NULL, NULL, NULL, NULL);		
	}
	while ( !GetAsyncKeyState(VK_ESCAPE) );
	VertexShader->Release();
	PixelShader->Release();
	VSBuffer->Release();
	PSBuffer->Release();
	d3dDevice->Release();
	d3d->Release();	
	return 0;
}
