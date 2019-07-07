// cl.exe checkerboard.cpp d3d11.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib

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
	"Texture2D<float4> pattern : register(t0);"
	"SamplerState state {Filter = MIN_MAG_LINEAR_MIP_POINT;};"
	"float4 PSMain(float4 vertex:SV_POSITION, float2 uv:TEXCOORD0) : SV_TARGET"
	"{"	
		"return pattern.Sample( state, uv );"
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
	ID3DBlob *VSblob;
	ID3DBlob *PSblob;
	ID3D11Texture2D *texture;
	ID3D11ShaderResourceView *srv;
	DXGI_SWAP_CHAIN_DESC sd = {{WIDTH, HEIGHT, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM, (DXGI_MODE_SCANLINE_ORDER)0, (DXGI_MODE_SCALING)0 }, {1, 0}, (1L << (1 + 4)) | (1L << (6 + 4)) | (1L << (0 + 4)), 1, hwnd, 1, DXGI_SWAP_EFFECT_SEQUENTIAL, 0};
	D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, 0, 0, 0, D3D11_SDK_VERSION, &sd, &surface, &device, 0, &context); 	
	surface->GetBuffer(0, __uuidof( ID3D11Resource), (void**)&image );
	device->CreateRenderTargetView(image, 0, &target);
	context->OMSetRenderTargets(1, &target, 0);
	D3D11_VIEWPORT vp = {0,0,WIDTH,HEIGHT,0.0f,1.0f};
	context->RSSetViewports(1, &vp);
	D3DCompile(&VertexShader, sizeof VertexShader, 0, 0, 0, "VSMain", "vs_5_0", 1 << 15, 0, &VSblob, 0);
	device->CreateVertexShader(VSblob->GetBufferPointer(), VSblob->GetBufferSize(), 0, &vs);
	D3DCompile(&PixelShader, sizeof PixelShader, 0, 0, 0, "PSMain", "ps_5_0", 1 << 15, 0, &PSblob, 0);
	device->CreatePixelShader(PSblob->GetBufferPointer(), PSblob->GetBufferSize(), 0, &ps);	
	context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
	int *pixels = new int[512*512];
	for(int i=0; i<512; i++)
	{
		for(int j=0; j<512; j++)
		{
			pixels[i*512+j] = ((i&32)==(j&32)) ? 0x00000000 : 0xffffffff;
		}
	}
	D3D11_SUBRESOURCE_DATA tsd = {(void *)pixels, 512*4, 512*512*4};
	D3D11_TEXTURE2D_DESC tdesc = {512, 512, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1,0}, D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE, 0, 0};
	device->CreateTexture2D(&tdesc, &tsd, &texture);   
	device->CreateShaderResourceView(texture, 0, &srv);
	context->PSSetShaderResources(0, 1, &srv);
	while (!exit)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT) exit = 1;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		context->VSSetShader(vs, 0, 0);
		context->PSSetShader(ps, 0, 0);
		context->Draw(6, 0);
		surface->Present(0, 0);
	}
	return 0;
}