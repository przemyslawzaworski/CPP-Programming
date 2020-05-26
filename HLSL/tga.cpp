// Loads uncompressed 32-bit RGBA TGA file and generates mipmaps.
// cl.exe tga.cpp d3d11.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib

#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <stdio.h>

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
	"Texture2D<float4> bricks : register(t0);"
	"Texture2D<float4> plasma : register(t1);"
	"SamplerState state {Filter = MIN_MAG_LINEAR_MIP_POINT;};"
	"float4 PSMain(float4 vertex:SV_POSITION, float2 uv:TEXCOORD0) : SV_TARGET"
	"{"	
		"return (uv.x < 0.5) ? bricks.SampleLevel( state, uv, 5 ) :  plasma.SampleLevel( state, uv, 3 );"
	"}"
};

ID3D11ShaderResourceView* LoadTexture (char* path, ID3D11Device* device, ID3D11DeviceContext* context)
{
	ID3D11Texture2D *texture;
	ID3D11ShaderResourceView *srv;
	FILE* file = fopen(path, "rb");
	unsigned char* header = new unsigned char[18];
	fread(header, sizeof( unsigned char ), 18, file);		
	unsigned short width = *(unsigned short*) &header[12];
	unsigned short height = *(unsigned short*) &header[14];	
	unsigned char* pixels = new unsigned char[width * height * 4];
	fread(pixels, sizeof( unsigned char ), width * height * 4, file);
	fclose(file);	
	D3D11_TEXTURE2D_DESC tdesc = {width, height, 0, 1, DXGI_FORMAT_B8G8R8A8_UNORM, {1,0}, D3D11_USAGE_DEFAULT, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0, D3D11_RESOURCE_MISC_GENERATE_MIPS};
	device->CreateTexture2D(&tdesc, NULL, &texture);
	context->UpdateSubresource(texture, 0, NULL, (void*)pixels, (unsigned int)(width * 4), 0);	
	D3D11_SHADER_RESOURCE_VIEW_DESC srvd = {DXGI_FORMAT_B8G8R8A8_UNORM, D3D11_SRV_DIMENSION_TEXTURE2D, {0u, -1u}};	
	device->CreateShaderResourceView(texture, &srvd, &srv);
	context->GenerateMips(srv);	
	delete[] header;
	delete[] pixels;
	texture->Release();
	return srv;
}

ID3D11ShaderResourceView* BlankTexture (ID3D11Device* device, ID3D11DeviceContext* context)
{
	ID3D11Texture2D *texture;
	ID3D11ShaderResourceView *srv;
	int *pixels = new int[16*16];
	for(int i=0; i<16; i++)
	{
		for(int j=0; j<16; j++)
		{
			pixels[i*16+j] = 0xffff0000;
		}
	}
	D3D11_SUBRESOURCE_DATA tsd = {(void *)pixels, 16*4, 16*16*4};
	D3D11_TEXTURE2D_DESC tdesc = {16, 16, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1,0}, D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE, 0, 0};
	device->CreateTexture2D(&tdesc, &tsd, &texture);   
	device->CreateShaderResourceView(texture, 0, &srv);
	delete[] pixels;
	texture->Release();
	return srv;	
}

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
	ID3D11ShaderResourceView* bricks = LoadTexture("bricks.tga", device, context);
	ID3D11ShaderResourceView* plasma = LoadTexture("plasma.tga", device, context);
	ID3D11ShaderResourceView* blank = BlankTexture(device, context);
	context->PSSetShaderResources(0, 1, &bricks);
	context->PSSetShaderResources(1, 1, &plasma);
	context->VSSetShader(vs, 0, 0);
	context->PSSetShader(ps, 0, 0);
	while (!exit)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT) exit = 1;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		if (GetAsyncKeyState(0x4F) && bricks == 0 && plasma == 0)  //O key
		{
			bricks = LoadTexture("bricks.tga", device, context);
			plasma = LoadTexture("plasma.tga", device, context);
			context->PSSetShaderResources(0, 1, &bricks);
			context->PSSetShaderResources(1, 1, &plasma);
		}
		if (GetAsyncKeyState(0x50))  //P key
		{
			context->PSSetShaderResources(0, 1, &blank);
			context->PSSetShaderResources(1, 1, &blank);
			if (plasma) {plasma->Release(); plasma = 0;}
			if (bricks) {bricks->Release(); bricks = 0;}
		}
		context->Draw(6, 0);
		surface->Present(0, 0);
	}
	vs->Release();
	ps->Release();
	VSblob->Release();
	PSblob->Release();
	image->Release();	
	target->Release();	
	surface->Release();
	context->Release();	
	device->Release();	
	return 0;
}