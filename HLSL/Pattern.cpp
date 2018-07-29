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
	"vs.3.0\n"
	"dcl_position v0\n"
	"dcl_position o0\n"
	"mov o0, v0\n";

static const char PixelShaderCode[] = \
	"ps.3.0\n"
	"def c0, 2, -1366, -768, 0.00520833349\n"
	"def c1, 10, 0, 1, 0\n"
	"def c2, 0.159154937, 0.5, 6.28318548, -3.14159274\n"
	"dcl vPos.xy\n"
	"mad r0.xy, vPos, c0.x, c0.yzzw\n"
	"mul r0.xy, r0, c0.w\n"
	"mul r0.z, r0_abs.y, r0_abs.x\n"
	"mad r0.z, r0.z, c2.x, c2.y\n"
	"frc r0.z, r0.z\n"
	"mad r0.z, r0.z, c2.z, c2.w\n"
	"sincos r1.x, r0.z\n"
	"mul r0.z, r1.x, c1.x\n"
	"lrp r1.x, r0.z, r0_abs.y, r0_abs.x\n"
	"add oC0.x, -r1_abs.x, c1.x\n"
	"mov oC0.yzw, c1.xyyz\n";

int main()
{
	LPDIRECT3DVERTEXSHADER9 VertexShader;
	LPDIRECT3DPIXELSHADER9 PixelShader;
	LPD3DXBUFFER VSBuffer, PSBuffer, PSDebug;	
	LPDIRECT3DDEVICE9 d3dDevice;	
	LPDIRECT3D9 d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS W = {1366,768,D3DFMT_A8R8G8B8,1,MSAA,0,SWAP,0,0,1,D3DFMT_D24S8,0,0,VSYNC};	
	W.hDeviceWindow = CreateWindow("static",0,WS_POPUP|WS_VISIBLE,0,0,1366,768,0,0,0,0);
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