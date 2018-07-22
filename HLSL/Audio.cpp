// g++ -s -o Audio.exe Audio.cpp  "-IC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Include" "-LC:\Program Files (x86)\Microsoft DirectX SDK (June 2010)\Lib\x86" -lwinmm -ld3d9 -ld3dx9
#define WIN32_LEAN_AND_MEAN
#define WIN32_EXTRA_LEAN
#define DLL D3DXSHADER_OPTIMIZATION_LEVEL3|D3DXSHADER_PREFER_FLOW_CONTROL
#define MSAA D3DMULTISAMPLE_NONE
#define VSYNC D3DPRESENT_INTERVAL_IMMEDIATE
#define SWAP D3DSWAPEFFECT_DISCARD
#include <windows.h>
#include <mmsystem.h>
#include <math.h>
#include <d3d9.h>
#include <d3dx9.h>

static const char VertexShaderCode[] = \
"float4 VSMain(float4 P:POSITION):POSITION {return P;};" ;

static const char PixelShaderCode[] = \
"float T : register(c0);"
"void PSMain(float2 U:VPOS, out float4 S:COLOR) "
"{"
	"U = U/float2(1920,1080);"
	"S = lerp(float4(0,sin(T)*0.5+0.5,0,1),0.5,step(U.x,0.5));"
"}" ;

int h[11] = {0x46464952,9172836,0x45564157,0x20746D66,16,WAVE_FORMAT_PCM|131072,44100,176400,1048580,0x61746164,9172800};
short m[9172822];

int f2i(float x) 
{
	if (x>=0x1.0p23) return x;
	return (unsigned int) (x+0.49999997f);
}

void audio(short *buffer)
{
	for (int i = 0; i<4586400; i++)
	{
		float t = (float)i / (float)44100;		
		float f = fmod(sinf(t*fmod(t,1.0f))*10.0f,1.0f);
		buffer[2*i+0] = f2i(f*32767.0f);
		buffer[2*i+1] = f2i(f*32767.0f);
	}
}

int main()
{
	LPDIRECT3DVERTEXSHADER9 VertexShader;
	LPDIRECT3DPIXELSHADER9 PixelShader;
	LPD3DXBUFFER VSBuffer, PSBuffer;	
	LPDIRECT3DDEVICE9 d3dDevice;	
	LPDIRECT3D9 d3d = Direct3DCreate9( D3D_SDK_VERSION );
	D3DPRESENT_PARAMETERS W = {1920,1080,D3DFMT_A8R8G8B8,1,MSAA,0,SWAP,0,0,1,D3DFMT_D24S8,0,0,VSYNC};	
	W.hDeviceWindow = CreateWindow("static",0,WS_POPUP|WS_VISIBLE,0,0,1920,1080,0,0,0,0);
	d3d->CreateDevice(0,D3DDEVTYPE_HAL,W.hDeviceWindow,D3DCREATE_HARDWARE_VERTEXPROCESSING,&W,&d3dDevice);
	ShowCursor(0);  
	D3DXCompileShader(VertexShaderCode,sizeof(VertexShaderCode),0,0,"VSMain","vs_3_0",DLL,&VSBuffer,0,0);
	D3DXCompileShader(PixelShaderCode,sizeof(PixelShaderCode),0,0,"PSMain","ps_3_0",DLL,&PSBuffer,0,0);
	d3dDevice->CreateVertexShader((DWORD*)VSBuffer->GetBufferPointer(), &VertexShader);
	d3dDevice->CreatePixelShader((DWORD*)PSBuffer->GetBufferPointer(), &PixelShader);
	d3dDevice->SetVertexShader(VertexShader);
	d3dDevice->SetPixelShader(PixelShader);
	d3dDevice->SetFVF(D3DFVF_XYZ);
	float quad[20] = {1,-1,0,1,0,-1,-1,0,0,0,1,1,0,1,1,-1,1,0,0,1}, S = GetTickCount()*0.001f;
	audio(m+22);
	memcpy(m,h,44);
	sndPlaySound((const char*)&m,SND_ASYNC|SND_MEMORY);	
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
	return 0;
}
