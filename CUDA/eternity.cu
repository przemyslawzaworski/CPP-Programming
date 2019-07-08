// nvcc -o eternity.exe eternity.cu -arch=sm_30 d3d11.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib

#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <cuda_d3d11_interop.h>

#define WIDTH 1280   //screen width
#define HEIGHT 720   //screen height
#define SIZE 1024   //texture size

surface<void, cudaSurfaceType2D> RenderSurface;

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

__device__ float smoothstep(float a, float b, float x)
{
	float t = fmaxf(0.0f, fminf((x - a)/(b - a), 1.0f));
	return t*t*(3.0f-(2.0f*t));
}

__global__ void mainImage (float iTime)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;
	float2 iResolution = make_float2((float)SIZE, (float)SIZE);
	float2 fragCoord = make_float2((float)x, (float)y);
	float2 uv = make_float2((2.0f * fragCoord.x / iResolution.x - 1.0f),(2.0f * fragCoord.y / iResolution.y - 1.0f));
	float L = sqrt(uv.x*uv.x+uv.y*uv.y)*4.0f;
	float K = atan2(uv.y, uv.x)+iTime;
	float X = fmod(sin(K*3.0f), cos(K*3.0f));
	float Y = fmod(cos(K*3.0f), sin(K*3.0f));
	float3 A = make_float3(X, X, 1.0f-X);
	float3 B = make_float3(Y+1.0f, Y+2.0f, Y+5.0f);
	float3 T = make_float3(1.0f-sin(iTime)*L, 1.0f-cos(iTime)*L, 1.0f-cos(iTime)*L);
	float3 color = make_float3(0.9f-smoothstep(A.x,B.x,T.x), 0.9f-smoothstep(A.y,B.y,T.y), 0.9f-smoothstep(A.z,B.z,T.z));
	uchar4 fragColor = make_uchar4(0.9f*color.x*255, 0.9f*color.y*255, 0.5f*color.z*255, 255);	
	surf2Dwrite(fragColor, RenderSurface, x*sizeof(uchar4), y, cudaBoundaryModeClamp);
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
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "CUDA DX11"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "CUDA DX11", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, WIDTH, HEIGHT, 0, 0, 0, 0);
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
	DXGI_SWAP_CHAIN_DESC sd = {{WIDTH, HEIGHT, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM, (DXGI_MODE_SCANLINE_ORDER)0, (DXGI_MODE_SCALING)0}, {1, 0}, (1L << (1 + 4)) | (1L << (6 + 4)) | (1L << (0 + 4)), 1, hwnd, 1, DXGI_SWAP_EFFECT_SEQUENTIAL, 0};
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
	D3D11_TEXTURE2D_DESC tdesc = {SIZE, SIZE, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM, {1,0}, D3D11_USAGE_DEFAULT,D3D11_BIND_SHADER_RESOURCE, 0, 0};
	device->CreateTexture2D(&tdesc, 0, &texture);   
	device->CreateShaderResourceView(texture, 0, &srv);
	context->PSSetShaderResources(0, 1, &srv);
	dim3 block(8, 8, 1);
	dim3 grid(SIZE/block.x, SIZE/block.y, 1);
	cudaGraphicsResource *resource;
	cudaArray *buffer;
	cudaGraphicsD3D11RegisterResource(&resource, texture, cudaGraphicsRegisterFlagsSurfaceLoadStore);
	cudaGraphicsMapResources(1, &resource, 0);
	cudaGraphicsSubResourceGetMappedArray(&buffer, resource, 0, 0);
	cudaBindSurfaceToArray(RenderSurface, buffer);	
	while (!exit)
	{
		while (PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if (msg.message == WM_QUIT) exit = 1;
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
		mainImage <<< grid, block >>> (GetTickCount()*0.001f);
		context->VSSetShader(vs, 0, 0);
		context->PSSetShader(ps, 0, 0);
		context->Draw(6, 0);
		surface->Present(0, 0);
	}
	return 0;
}