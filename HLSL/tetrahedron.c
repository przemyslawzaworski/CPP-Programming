// cl.exe tetrahedron.c d3d11.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib

#include <Windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>

#define WIDTH 1280 
#define HEIGHT 720

const unsigned char ComputeShader[] =
{
	"RWTexture2D<float4> image : register (u0);"

	"cbuffer Constants : register(b0)"
	"{"
		"float iTime;"
	"};"

	"float3x3 rotationX(float x) "
	"{"
		"return float3x3(1.0,0.0,0.0,0.0,cos(x),sin(x),0.0,-sin(x),cos(x));"
	"}"

	"float3x3 rotationY(float y) "
	"{"
		"return float3x3(cos(y),0.0,-sin(y),0.0,1.0,0.0,sin(y),0.0,cos(y));"
	"}"

	"bool InsideTetrahedron(float3 a, float3 b, float3 c, float3 d, float3 p, out float3 color)"
	"{"
		"p=mul(rotationY(iTime),mul(rotationX(iTime),p));"
		"float3 vap = p - a;"
		"float3 vbp = p - b;"
		"float3 vab = b - a;"
		"float3 vac = c - a;"
		"float3 vad = d - a;"
		"float3 vbc = c - b;"
		"float3 vbd = d - b;"
		"float va6 = dot(vbp, cross(vbd, vbc));"
		"float vb6 = dot(vap, cross(vac, vad));"
		"float vc6 = dot(vap, cross(vad, vab));"
		"float vd6 = dot(vap, cross(vab, vac));"
		"float v6 = 1.0 / dot(vab, cross(vac, vad));"
		"float4 k =  float4(va6*v6, vb6*v6, vc6*v6, vd6*v6);"
		"if ((k.x >= 0.0) && (k.x <= 1.0) && (k.y >= 0.0) && (k.y <= 1.0) && (k.z >= 0.0) && (k.z <= 1.0) && (k.w >= 0.0) && (k.w <= 1.0))"
		"{"
			"color = k.rgb;"
			"return true;"
		"}"
		"else"
		"{"
			"color = float3(0.0,0.0,0.0);"
			"return false;"
		"}"          
	"}"

	"float4 raymarch (float3 ro, float3 rd)"
	"{"
		"float3 color = 0..xxx;"
		"for (int i = 0; i < 512; i++)"
		"{"
			"bool hit = InsideTetrahedron(float3(0.943, 0, -0.333 ),float3( -0.471, 0.816, -0.333), float3( -0.471, -0.816, -0.333), float3(0, 0, 1 ),ro,color);"
			"if (hit) return float4(color,1.0);"
			"ro += rd * 0.01;"
		"}"
		"return float4(0,0,0,1);"
	"}"

	"[numthreads(8, 8, 1)]"
	"void main (uint3 id : SV_DispatchThreadID)"
	"{"
		"float2 uv = (2.*id.xy - float2(1280, 720)) / 720.0;"
		"image[id.xy] = raymarch(float3(0,0.0,-2.5), normalize(float3(uv,2.)));"
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
	ID3D11Buffer *buffer;      
	ID3D11UnorderedAccessView *uav; 
	ID3D11Texture2D *image;	
	ID3D11ComputeShader *shader;
	D3D11_MAPPED_SUBRESOURCE resource;	
	DXGI_SWAP_CHAIN_DESC sd = {{WIDTH, HEIGHT, 0, 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 0 }, {1, 0}, (1L << (1 + 4)) | (1L << (6 + 4)) | (1L << (0 + 4)), 1, hwnd, 1, 1, 0};
	D3D11CreateDeviceAndSwapChain(0, D3D_DRIVER_TYPE_HARDWARE, 0, 0, 0, 0, D3D11_SDK_VERSION, &sd, &surface, &device, 0, &context);
	surface->lpVtbl->GetDesc(surface, &sd); 	
	surface->lpVtbl->GetBuffer(surface, 0, (REFIID) &IID_ID3D11Texture2D, ( LPVOID* )&image );
	D3D11_BUFFER_DESC desc = {16, D3D11_USAGE_DYNAMIC, D3D11_BIND_CONSTANT_BUFFER, D3D11_CPU_ACCESS_WRITE, 0, 0};	
	device->lpVtbl->CreateBuffer(device, &desc, NULL, &buffer);
	device->lpVtbl->CreateUnorderedAccessView(device,(ID3D11Resource*)image, NULL, &uav );	
	ID3DBlob* blob;
	D3DCompile(&ComputeShader, sizeof ComputeShader, 0, 0, 0, "main", "cs_5_0", 1 << 15, 0, &blob, 0);		
	device->lpVtbl->CreateComputeShader(device, blob->lpVtbl->GetBufferPointer(blob), blob->lpVtbl->GetBufferSize(blob), NULL, &shader);
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
		context->lpVtbl->CSSetShader(context, shader, NULL, 0 );
		context->lpVtbl->CSSetUnorderedAccessViews(context, 0, 1, &uav, NULL );
		context->lpVtbl->CSSetConstantBuffers(context, 0, 1, &buffer );			
		context->lpVtbl->Dispatch(context, WIDTH  / 8, HEIGHT  / 8, 1 );
		surface->lpVtbl->Present( surface, 0, 0 );
	}
	context->lpVtbl->ClearState(context);
	device->lpVtbl->Release(device);
	surface->lpVtbl->Release(surface);	 
	image->lpVtbl->Release(image);	
	buffer->lpVtbl->Release(buffer);
	uav->lpVtbl->Release(uav);
	return 0; 
}