// cl.exe particles.c d3d11.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib

#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#define WIDTH 1280 
#define HEIGHT 720

const unsigned char VertexShader[] =
{
	"static const float3 vertices[6] = {float3(1,-1,0),float3(-1,-1,0),float3(1,1,0), float3(-1,-1,0),float3(-1,1,0),float3(1,1,0)};"
	"static const float2 uvs[6] = {float2(1,0),float2(0,0),float2(1,1), float2(0,0),float2(0,1),float2(1,1)};"
	"void VSMain(out float4 vertex:SV_POSITION, out float2 uv:TEXCOORD0, in uint id:SV_VertexID)"
	"{"	
		"uv = uvs[id];"
		"vertex = float4(vertices[id], 1);"
	"}"
};

const unsigned char PixelShader[] =
{
	"cbuffer Constants : register(b0)"
	"{"
		"float iTime;"
	"};"
	"float3 surface (float2 uv)"
	"{"
		"float2 k = 0;"
		"for (float i=0.0;i<64.0;i++)"
		"{"
			"float2 q = float2(i*127.1+i*311.7,i*269.5+i*183.3);"
			"float2 h = frac(sin(q)*43758.5453);"
			"float2 p = cos(h*iTime);"
			"float d = length(uv-p);"
			"k+=(1.0-step(0.06,d))*h;"
		"}"
		"return float3(0.0,k);"
	"}"	
	"float4 PSMain(float4 vertex:SV_POSITION, float2 uv:TEXCOORD0) : SV_TARGET"
	"{"	
		"float2 p = float2(2.0*uv-1.0)/(1.0-uv.y);"
		"float3 c = 0;"
		"float2 d = (float2(0.0,-1.0)-p)/float(80);"
		"float w = 1.0;"
		"float2 s = p;"
		"for( int i=0; i<80; i++ )"
		"{"
			"float3 res = surface(s);"
			"c += w*smoothstep(0.0, 1.0, res);"
			"w *= 0.97;"
			"s += d;"
		"}"
		"c = c * 8.0 / float(80);"
		"return float4(c, 1.0);"
	"}"
};

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if ((uMsg == WM_KEYUP && wParam == VK_ESCAPE) || uMsg==WM_CLOSE || uMsg==WM_DESTROY)
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "DirectX 11"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "DirectX 11", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, WIDTH, HEIGHT, 0, 0, 0, 0);	
	ID3D11Device *device;
	IDXGISwapChain *surface;
	ID3D11DeviceContext *context;     
	ID3D11Resource *image;	
	ID3D11RenderTargetView *target;
	ID3D11VertexShader *vs;
	ID3D11PixelShader *ps;
	ID3DBlob* VSblob;
	ID3DBlob* PSblob;	
	ID3D11Buffer *buffer; 
	D3D11_MAPPED_SUBRESOURCE resource;
	DXGI_SWAP_CHAIN_DESC sd = {{WIDTH, HEIGHT, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0 }, {1, 0}, (1L << (1 + 4)) | (1L << (6 + 4)) | (1L << (0 + 4)), 1, hwnd, 1, 1, 0};
	D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, 0, 0, 0, D3D11_SDK_VERSION, &sd, &surface, &device, 0, &context); 	
	surface->lpVtbl->GetBuffer(surface, 0, (REFIID) &IID_ID3D11Resource, ( LPVOID* )&image );
	D3D11_BUFFER_DESC desc = {16, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0};	
	device->lpVtbl->CreateBuffer(device, &desc, 0, &buffer);
	device->lpVtbl->CreateRenderTargetView(device, image, 0, &target);
	context->lpVtbl->OMSetRenderTargets(context,1, &target, 0);
	D3D11_VIEWPORT vp = {0,0,WIDTH,HEIGHT,0.0f,1.0f};
	context->lpVtbl->RSSetViewports(context,1, &vp);
	D3DCompile(&VertexShader, sizeof VertexShader, 0, 0, 0, "VSMain", "vs_5_0", 1 << 15, 0, &VSblob, 0);		
	device->lpVtbl->CreateVertexShader(device, VSblob->lpVtbl->GetBufferPointer(VSblob), VSblob->lpVtbl->GetBufferSize(VSblob), 0, &vs);
	D3DCompile(&PixelShader, sizeof PixelShader, 0, 0, 0, "PSMain", "ps_5_0", 1 << 15, 0, &PSblob, 0);		
	device->lpVtbl->CreatePixelShader(device, PSblob->lpVtbl->GetBufferPointer(PSblob), PSblob->lpVtbl->GetBufferSize(PSblob), 0, &ps);	
	context->lpVtbl->IASetPrimitiveTopology(context,D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	while (!exit)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT) exit = 1;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		context->lpVtbl->Map(context,(ID3D11Resource*)buffer, 0, D3D11_MAP_WRITE_DISCARD, 0,  &resource);
		float time[] = {GetTickCount() * 0.001f};
		memcpy(resource.pData, time, sizeof(float));		
		context->lpVtbl->Unmap(context, (ID3D11Resource *)buffer, 0);			
		context->lpVtbl->VSSetShader(context, vs, 0, 0 );
		context->lpVtbl->PSSetShader(context, ps, 0, 0 );
		context->lpVtbl->PSSetConstantBuffers(context, 0, 1, &buffer );	
		context->lpVtbl->Draw(context, 6, 0);		
		surface->lpVtbl->Present(surface, 0, 0 );
	}
	return 0;
}