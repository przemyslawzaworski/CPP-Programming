//g++ -s -o Helix.exe Helix.cpp  "-IC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include" "-LC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Lib\x86" -ld3d9 -ld3dx9
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#define DLL D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXSHADER_PREFER_FLOW_CONTROL
#define MSAA D3DMULTISAMPLE_NONE
#define VSYNC D3DPRESENT_INTERVAL_IMMEDIATE
#define SWAP D3DSWAPEFFECT_DISCARD
#include <windows.h>
#include <d3d9.h>
#include <d3dx9.h>
#include <stdio.h>

static const char VertexShaderCode[] = \
"float4 VSMain(float4 P:POSITION):POSITION {return P;};" ;

static const char PixelShaderCode[] = \
"uniform extern float T : register(c0);"
"float mod(float x, float y)"
"{"
	"return x - y * floor(x/y);"
"}"
"float map (float3 q)"
"{"
	"float r = 20.0;"
	"float a = atan2(q.z,q.x); "
	"q.x = length(q.xz)-r; " 
	"q.y = mod(q.y-a*r/6.28,r)-r*0.5;"
	"q.z = r*a;"
	"float l = length(q.xy);"
	"float d = sin(atan2(q.y,q.x)-q.z);"
	"return length(float2(l-4.0,d)) - 0.5;"
"}"
"float4 raymarch (float3 ro, float3 rd)"
"{"
	"float4 n = float4(0,0,0,1); "
	"for (int i=0;i<128;i++)"
	"{"
		"float t = map (ro);"
		"if (t<0.001)"
		"{"
			"float c = pow(1.0-float(i)/float(128),2.0);"
			"n = float4(c,c,c,1.0); break;" 
		"}"
		"ro+=t*rd;"
	"}"
	"return n;"
"}"
"void PSMain(float2 U:VPOS, out float4 S:COLOR) "
"{"
	"float2 UV = (2.0*U-float2(1920,1080))/1080.0;"
	"float3 ro = float3(0.0,T*3.0,-40.0);"
	"float3 rd = normalize(float3(UV,2.0));	"		
	"S = raymarch(ro,rd);"
"}" ;

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
	D3DXCompileShader(VertexShaderCode,sizeof(VertexShaderCode),0,0,"VSMain","vs_3_0",DLL,&VSBuffer,0,0);
	D3DXCompileShader(PixelShaderCode,sizeof(PixelShaderCode),0,0,"PSMain","ps_3_0",DLL,&PSBuffer,&PSDebug,0);
	if (PSDebug)
	{
		FILE *file = fopen ("debug.log","a");
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
	float quad[20] = {1,-1,0,1,0,-1,-1,0,0,0,1,1,0,1,1,-1,1,0,0,1}, S = GetTickCount()*0.001f;
	do 
	{
		d3dDevice->BeginScene();
		float timer[1] = {GetTickCount()*0.001f-S};
		d3dDevice->SetPixelShaderConstantF(0, timer, 1);
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
