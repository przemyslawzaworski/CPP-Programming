#include <d3dx9.h>

static const char PixelShaderCode[] = \
	"ps_3_0\n"
	"def c0, 2, -1366, -768, 0.00130208337\n"
	"def c1, 0.7, 0, 0, 0\n"
	"def c2, 0, 0.5, -1, 0\n"
	"dcl vPos.xy\n"
	"mad r0.xy, vPos, c0.x, c0.yzzw\n"
	"mul r0.xy, r0, c0.w\n"
	"dp2add r0.x, r0, r0, c2.x\n"
	"rsq r0.x, r0.x\n"
	"rcp r0.x, r0.x\n"
	"add r0.y, -r0.x, c2.y\n"
	"cmp r0.y, r0.y, c2.z, c2.w\n"
	"add r0.x, r0.y, -r0.x\n"
	"add r0.x, r0.x, c1.x\n"
	"cmp oC0.z, r0.x, -c2.z, -c2.w\n"
	"mov oC0.xyw, -c2.wwzz\n";

int main()
{
	LPDIRECT3DPIXELSHADER9 PixelShader;
	LPD3DXBUFFER PSBuffer;	
	LPDIRECT3DDEVICE9 d3dDevice;	
	LPDIRECT3D9 d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS W = {1366,768,D3DFMT_A8R8G8B8,1,D3DMULTISAMPLE_NONE,0,_D3DSWAPEFFECT(1),0,0,1,D3DFMT_D24S8,0,0,0x80000000L};	
	W.hDeviceWindow = CreateWindow("static",0,WS_POPUP|WS_VISIBLE,0,0,1366,768,0,0,0,0);
	d3d->CreateDevice(0,D3DDEVTYPE_HAL,W.hDeviceWindow,0x00000040L,&W,&d3dDevice);
	ShowCursor(0);  
	D3DXAssembleShader(PixelShaderCode,sizeof(PixelShaderCode), 0,0, (1 << 16), &PSBuffer, 0);
	d3dDevice->CreatePixelShader((DWORD*)PSBuffer->GetBufferPointer(), &PixelShader);
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
	return 0;
}