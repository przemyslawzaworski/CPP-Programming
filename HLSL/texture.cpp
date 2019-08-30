// cl.exe texture.cpp d3d12.lib dxgi.lib dxguid.lib user32.lib kernel32.lib gdi32.lib d3dcompiler.lib
#include <Windows.h>
#include <d3d12.h>
#include <math.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <stdio.h>

#define ScreenWidth 1920.0f
#define ScreenHeight 1080.0f
#define FieldOfView 60.0f
#define NearClip 0.01f
#define FarClip 1000.0f
#define VerticalSync 0

static const int QUEUE_SLOT_COUNT = 3;
IDXGISwapChain* mSwapChain;
ID3D12Device* mDevice;
ID3D12Resource* renderTargets[QUEUE_SLOT_COUNT];
ID3D12CommandQueue* mCommandQueue;
HANDLE frameFenceEvents[QUEUE_SLOT_COUNT];
ID3D12Fence* mFrameFences[QUEUE_SLOT_COUNT];
int mFenceValues[QUEUE_SLOT_COUNT];
ID3D12DescriptorHeap* mrtDescriptorHeap;
ID3D12RootSignature* mRootSignature;
ID3D12PipelineState* mPSO;
ID3D12CommandAllocator* mCommandAllocator[QUEUE_SLOT_COUNT];
ID3D12GraphicsCommandList* mCommandList[QUEUE_SLOT_COUNT];
int mCurrentBackBuffer = 0;
ID3D12Resource* mUploadBuffer;
ID3D12Resource* mVertexBuffer;
ID3D12Resource* mIndexBuffer;
ID3D12Resource*	mImage;
ID3D12Resource*	mUploadImage;
ID3D12Resource* mConstantBuffers[QUEUE_SLOT_COUNT];
ID3D12DescriptorHeap* msrvDescriptorHeap;

float CameraRotationMatrix[4][4], ViewMatrix[4][4], ProjectionViewMatrix[4][4], MVP[4][4], iMouse[2] = {0.0f,0.0f};

const char Shaders[] =
	"cbuffer PerFrameConstants : register (b0) "
	"{"
		"float4x4 MVP;"
	"}"
	"float4 VSMain(float4 vertex:POSITION, inout float2 uv:TEXCOORD0) : SV_POSITION"
	"{"	
		"return mul(MVP, vertex);"
	"}"
	"Texture2D<float4> pattern : register(t0);"
	"SamplerState state : register(s0) ;"
	"float4 PSMain(float4 vertex:SV_POSITION, float2 uv:TEXCOORD0) : SV_TARGET"
	"{"	
		"return pattern.Sample( state, uv );"
	"}";

float CameraTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,-1.0f,-5.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationYMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationXMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,1.0f,0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ProjectionMatrix[4][4] = 
{
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,0.0f,0.0f,
	0.0f,0.0f,-1.0f,0.0f
};

float deg2rad(float x) 
{
	return (x * 3.14159265358979323846f / 180.0f);
}

void Mul(float mat1[][4], float mat2[][4], float res[][4])
{
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			res[i][j] = 0;
			for (int k = 0; k < 4; k++) 
			{
				res[i][j] += mat1[i][k]*mat2[k][j];
			}
		}
	}
}

void Inverse( float input[4][4], float k[4][4])
{
	float invOut[16];
	float m[16] = 
	{
		input[0][0],input[0][1],input[0][2],input[0][3],
		input[1][0],input[1][1],input[1][2],input[1][3],
		input[2][0],input[2][1],input[2][2],input[2][3],
		input[3][0],input[3][1],input[3][2],input[3][3]
	};
	float inv[16], det;
	int i;
	inv[0]  =  m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
	inv[4]  = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
	inv[8]  =  m[4] *m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7] *m[9];
	inv[12] = -m[4] *m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6] *m[9];
	inv[1]  = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
	inv[5]  =  m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
	inv[9]  = -m[0] *m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3] *m[9];
	inv[13] =  m[0] *m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2] *m[9];
	inv[2]  =  m[1] *m[6]*m[15]-m[1] *m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2] *m[7]-m[13]*m[3] *m[6];
	inv[6]  = -m[0] *m[6]*m[15]+m[0] *m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2] *m[7]+m[12]*m[3] *m[6];
	inv[10] =  m[0] *m[5]*m[15]-m[0] *m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1] *m[7]-m[12]*m[3] *m[5];
	inv[14] = -m[0] *m[5]*m[14]+m[0] *m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1] *m[6]+m[12]*m[2] *m[5];
	inv[3]  = -m[1] *m[6]*m[11]+m[1] *m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10] -m[9]*m[2] *m[7] +m[9]*m[3] *m[6];
	inv[7]  =  m[0] *m[6]*m[11]-m[0] *m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10] +m[8]*m[2] *m[7] -m[8]*m[3] *m[6];
	inv[11] = -m[0] *m[5]*m[11]+m[0] *m[7]*m[9] +m[4]*m[1]*m[11]-m[4]*m[3] *m[9] -m[8]*m[1] *m[7] +m[8]*m[3] *m[5];
	inv[15] =  m[0] *m[5]*m[10]-m[0] *m[6]*m[9] -m[4]*m[1]*m[10]+m[4]*m[2] *m[9] +m[8]*m[1] *m[6] -m[8]*m[2] *m[5];
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	det = 1.0 / det;
	for (i = 0; i < 16; i++) invOut[i] = inv[i] * det;	
	k[0][0] = invOut[0];  k[0][1] = invOut[1];  k[0][2] = invOut[2];  k[0][3] = invOut[3];
	k[1][0] = invOut[4];  k[1][1] = invOut[5];  k[1][2] = invOut[6];  k[1][3] = invOut[7];
	k[2][0] = invOut[8];  k[2][1] = invOut[9];  k[2][2] = invOut[10]; k[2][3] = invOut[11];
	k[3][0] = invOut[12]; k[3][1] = invOut[13]; k[3][2] = invOut[14]; k[3][3] = invOut[15];  
}

void MouseLook()
{	
	POINT point;
	int mx = (int)ScreenWidth  >> 1;
	int my = (int)ScreenHeight >> 1;
	GetCursorPos(&point);
	if( (point.x == mx) && (point.y == my) ) return;
	SetCursorPos(mx, my);	
	float deltaZ = (float)((mx - point.x)) ;
	float deltaX = (float)((my - point.y)) ;
	if (deltaX>0.0f) iMouse[0]+=1.5f; 
	if (deltaX<0.0f) iMouse[0]-=1.5f; 
	if (deltaZ>0.0f) iMouse[1]+=1.5f; 
	if (deltaZ<0.0f) iMouse[1]-=1.5f; 
	CameraRotationXMatrix[1][1] = cos(deg2rad(iMouse[0]));
	CameraRotationXMatrix[1][2] = (-1.0f)*sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][1] = sin(deg2rad(iMouse[0]));
	CameraRotationXMatrix[2][2] = cos(deg2rad(iMouse[0]));
	CameraRotationYMatrix[0][0] = cos(deg2rad(iMouse[1]));
	CameraRotationYMatrix[0][2] = sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][0] = (-1.0f)*sin(deg2rad(iMouse[1]));
	CameraRotationYMatrix[2][2] = cos(deg2rad(iMouse[1]));
}

void KeyboardMovement()
{
	float dx = 0.0f;
	float dz = 0.0f;
	if (GetAsyncKeyState(0x57)) dz =  16.0f;
	if (GetAsyncKeyState(0x53)) dz = -16.0f ;
	if (GetAsyncKeyState(0x44)) dx =  16.0f;
	if (GetAsyncKeyState(0x41)) dx = -16.0f ;
	CameraTranslationMatrix[0][3] += (-dz * ViewMatrix[2][0] + dx * ViewMatrix[0][0]) * 0.001f;
	CameraTranslationMatrix[1][3] += (-dz * ViewMatrix[2][1] + dx * ViewMatrix[1][0]) * 0.001f;  //comment to enable FPP
	CameraTranslationMatrix[2][3] += (-dz * ViewMatrix[2][2] + dx * ViewMatrix[2][0]) * 0.001f;	
}

void WaitForFence(ID3D12Fence* fence, UINT64 completionValue, HANDLE waitEvent)
{
	if (fence->GetCompletedValue() < completionValue) 
	{
		fence->SetEventOnCompletion(completionValue, waitEvent);
		WaitForSingleObject(waitEvent, INFINITE);
	}
}

UINT64 UpdateSubResources(ID3D12GraphicsCommandList* pCmdList, ID3D12Resource* pDestinationResource, ID3D12Resource* pIntermediate, UINT64 Offset, UINT FirstSubresource, UINT NumSubresources, D3D12_SUBRESOURCE_DATA* pSrcData)
{
	UINT64 RequiredSize = 0;
	UINT64 MemToAlloc = (UINT64)(sizeof(D3D12_PLACED_SUBRESOURCE_FOOTPRINT) + sizeof(UINT) + sizeof(UINT64)) * NumSubresources;
	void* pMem = HeapAlloc(GetProcessHeap(), 0, (SIZE_T)(MemToAlloc));
	D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts = (D3D12_PLACED_SUBRESOURCE_FOOTPRINT*)(pMem);
	UINT64* pRowSizesInBytes = (UINT64*)(pLayouts + NumSubresources);
	UINT* pNumRows = (UINT*)(pRowSizesInBytes + NumSubresources);
	D3D12_RESOURCE_DESC Desc = pDestinationResource->GetDesc();
	ID3D12Device* pDevice;
	pDestinationResource->GetDevice(__uuidof(*pDevice), (void**)(&pDevice));
	pDevice->GetCopyableFootprints(&Desc, FirstSubresource, NumSubresources, Offset, pLayouts, pNumRows, pRowSizesInBytes, &RequiredSize);
	pDevice->Release();		
	D3D12_RESOURCE_DESC DestinationDesc = pDestinationResource->GetDesc();
	unsigned char* pData;
	pIntermediate->Map(0, 0, (void**)(&pData));
	for (UINT i = 0; i < NumSubresources; ++i)
	{
		if (pRowSizesInBytes[i] > (SIZE_T)-1) return 0;
		D3D12_MEMCPY_DEST DestData = { pData + pLayouts[i].Offset, pLayouts[i].Footprint.RowPitch, pLayouts[i].Footprint.RowPitch * pNumRows[i] };
		D3D12_MEMCPY_DEST* DestDataPointer = &DestData;
		D3D12_SUBRESOURCE_DATA* pSrcDataPointer = &pSrcData[i];
		for (UINT z = 0; z < pLayouts[i].Footprint.Depth; ++z)
		{
			unsigned char* pDestSlice = (unsigned char*)(DestDataPointer->pData) + DestDataPointer->SlicePitch * z;
			const unsigned char* pSrcSlice = (unsigned char*)(pSrcDataPointer->pData) + pSrcDataPointer->SlicePitch * z;
			for (UINT y = 0; y < pNumRows[i]; ++y)
			{
				memcpy(pDestSlice + DestDataPointer->RowPitch * y,pSrcSlice + pSrcDataPointer->RowPitch * y,(SIZE_T)pRowSizesInBytes[i]);
			}
		}
	}
	pIntermediate->Unmap(0, 0);
	if (DestinationDesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
	{
		D3D12_BOX SrcBox;
		ZeroMemory(&SrcBox, sizeof(SrcBox));
		SrcBox.left = UINT(pLayouts[0].Offset);
		SrcBox.right = UINT(pLayouts[0].Offset + pLayouts[0].Footprint.Width);
		pCmdList->CopyBufferRegion(pDestinationResource, 0, pIntermediate, pLayouts[0].Offset, pLayouts[0].Footprint.Width);
	}
	else
	{
		for (UINT i = 0; i < NumSubresources; ++i)
		{
			D3D12_TEXTURE_COPY_LOCATION Dst;
			ZeroMemory(&Dst, sizeof(Dst));
			Dst.pResource = pDestinationResource;
			Dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
			Dst.SubresourceIndex = i + FirstSubresource;
			D3D12_TEXTURE_COPY_LOCATION Src;
			ZeroMemory(&Src, sizeof(Src));
			Src.pResource = pIntermediate;
			Src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
			Src.PlacedFootprint = pLayouts[i];
			pCmdList->CopyTextureRegion(&Dst, 0, 0, 0, &Src, 0);
		}
	}
	HeapFree(GetProcessHeap(), 0, pMem);
	return RequiredSize;
}

void LoadTexture(char* filename, ID3D12GraphicsCommandList* uploadCommandList, ID3D12Device* mDevice, ID3D12Resource* image, ID3D12Resource* uploadImage, ID3D12DescriptorHeap* srvheap)
{	
	int width, height; 
	char buffer[128]; 
	FILE *file = fopen(filename, "rb");
	fgets(buffer, sizeof(buffer), file);
	do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
	sscanf (buffer, "%d %d", &width, &height);
	do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
	int size = width * height * 4 * sizeof(unsigned char);
	unsigned char *pixels  = (unsigned char *)malloc(size);
	for (int i = 0; i < size; i++) 
	{
		pixels[i] = ((i % 4) < 3 ) ? (unsigned char) fgetc(file) : (unsigned char) 255 ;
	}
	fclose(file);
	static D3D12_HEAP_PROPERTIES mDefaultHeap = {D3D12_HEAP_TYPE_DEFAULT,(D3D12_CPU_PAGE_PROPERTY)0,(D3D12_MEMORY_POOL)0,0,0};
	D3D12_RESOURCE_DESC mTexture;
	ZeroMemory(&mTexture, sizeof(mTexture));
	mTexture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
	mTexture.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
	mTexture.Width = width;
	mTexture.Height = height;
	mTexture.DepthOrArraySize = 1;
	mTexture.MipLevels = 1;
	mTexture.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	mTexture.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
	mTexture.SampleDesc.Count = 1;
	mDevice->CreateCommittedResource(&mDefaultHeap,D3D12_HEAP_FLAG_NONE,&mTexture,D3D12_RESOURCE_STATE_COPY_DEST,0,IID_PPV_ARGS(&image));
	static D3D12_HEAP_PROPERTIES mUploadHeap = {D3D12_HEAP_TYPE_UPLOAD,(D3D12_CPU_PAGE_PROPERTY)0,(D3D12_MEMORY_POOL)0,0,0};
	D3D12_RESOURCE_DESC Desc = image->GetDesc();
	UINT64 RequiredSize = 0;
	mDevice->GetCopyableFootprints(&Desc, 0, 1, 0, 0, 0, 0, &RequiredSize);
	D3D12_RESOURCE_DESC	mBuffer;
	ZeroMemory(&mBuffer, sizeof(mBuffer));
	mBuffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
	mBuffer.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
	mBuffer.Width = RequiredSize;
	mBuffer.Height = 1;
	mBuffer.DepthOrArraySize = 1;
	mBuffer.MipLevels = 1;
	mBuffer.Format = DXGI_FORMAT_UNKNOWN;
	mBuffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
	mBuffer.SampleDesc.Count = 1;
	mDevice->CreateCommittedResource(&mUploadHeap,D3D12_HEAP_FLAG_NONE,&mBuffer,D3D12_RESOURCE_STATE_GENERIC_READ,0,IID_PPV_ARGS(&uploadImage));
	D3D12_SUBRESOURCE_DATA srcData = {pixels, width * 4, width * height * 4};
	UpdateSubResources(uploadCommandList, image, uploadImage, 0, 0, 1, &srcData);	
	D3D12_RESOURCE_BARRIER transition;
	ZeroMemory(&transition, sizeof(transition));
	transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
	transition.Transition.pResource = image;
	transition.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
	transition.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
	transition.Transition.StateAfter = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;	
	uploadCommandList->ResourceBarrier(1, &transition);
	D3D12_SHADER_RESOURCE_VIEW_DESC srv;
	ZeroMemory(&srv, sizeof(srv));
	srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
	srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
	srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	srv.Texture2D.MipLevels = 1;
	srv.Texture2D.MostDetailedMip = 0;
	srv.Texture2D.ResourceMinLODClamp = 0.0f; 
	mDevice->CreateShaderResourceView(image, &srv, srvheap->GetCPUDescriptorHandleForHeapStart());
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

int WinMain (_In_ HINSTANCE, _In_opt_ HINSTANCE, _In_ LPSTR, _In_ int )
{
	ShowCursor(0);
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "DirectX 12"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "DirectX 12", WS_VISIBLE|WS_POPUP, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0);
	ProjectionMatrix[0][0] = ((1.0f/tan(deg2rad(FieldOfView/2.0f)))/(ScreenWidth/ScreenHeight));
	ProjectionMatrix[1][1] = (1.0f/tan(deg2rad(FieldOfView/2.0f)));
	ProjectionMatrix[2][2] = (-1.0f)* (FarClip+NearClip)/(FarClip-NearClip);
	ProjectionMatrix[2][3] = (-1.0f)*(2.0f*FarClip*NearClip)/(FarClip-NearClip);	
	DXGI_SWAP_CHAIN_DESC scd;
	ZeroMemory(&scd, sizeof(scd));
	scd.BufferDesc.RefreshRate.Numerator = 0;
	scd.BufferDesc.RefreshRate.Denominator = 0;	
	scd.BufferCount = QUEUE_SLOT_COUNT;
	scd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;	
	scd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
	scd.BufferDesc.Width = ScreenWidth;
	scd.BufferDesc.Height = ScreenHeight;
	scd.OutputWindow = hwnd;
	scd.SampleDesc.Count = 1;
	scd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
	scd.Windowed = false;		
	D3D12CreateDevice(0, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&mDevice));
	D3D12_COMMAND_QUEUE_DESC queue = {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};	
	mDevice->CreateCommandQueue(&queue, IID_PPV_ARGS(&mCommandQueue));
	IDXGIFactory4* dxgiFactory;
	CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory));
	dxgiFactory->CreateSwapChain(mCommandQueue, &scd, &mSwapChain);
	int mrtvDescriptorIncrSize = mDevice->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	int mCurrentFenceValue = 1;
	D3D12_DESCRIPTOR_HEAP_DESC heap = {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, QUEUE_SLOT_COUNT, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
	mDevice->CreateDescriptorHeap(&heap, IID_PPV_ARGS(&mrtDescriptorHeap));
	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = {mrtDescriptorHeap->GetCPUDescriptorHandleForHeapStart()};
	for (int i = 0; i < QUEUE_SLOT_COUNT; ++i) 
	{
		frameFenceEvents[i] = CreateEvent(0, 0, 0, 0);
		mFenceValues[i] = 0;
		mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&mFrameFences[i]));
		mSwapChain->GetBuffer(i, IID_PPV_ARGS(&renderTargets[i]));
		D3D12_RENDER_TARGET_VIEW_DESC viewDesc = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, D3D12_RTV_DIMENSION_TEXTURE2D, 0};
		mDevice->CreateRenderTargetView(renderTargets[i], &viewDesc,rtvHandle);
		rtvHandle.ptr += mrtvDescriptorIncrSize;
		mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&mCommandAllocator[i]));
		mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,mCommandAllocator[i], 0,IID_PPV_ARGS(&mCommandList[i]));
		mCommandList[i]->Close();		
	}	
	D3D12_RECT mRectScissor = { 0, 0, (LONG)ScreenWidth, (LONG)ScreenHeight };
	D3D12_VIEWPORT mViewport = { 0.0f, 0.0f, ScreenWidth, ScreenHeight, 0.0f, 1.0f};
	ID3D12Fence* uploadFence;
	mDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&uploadFence));
	ID3D12CommandAllocator* uploadCommandAllocator;
	mDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&uploadCommandAllocator));
	ID3D12GraphicsCommandList* uploadCommandList;
	mDevice->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,uploadCommandAllocator, 0,IID_PPV_ARGS(&uploadCommandList));
	D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
	mDevice->CreateDescriptorHeap(&descriptorHeapDesc, IID_PPV_ARGS(&msrvDescriptorHeap));	
	D3D12_ROOT_PARAMETER parameters[2];
	D3D12_DESCRIPTOR_RANGE  descriptorTableRanges; 
	descriptorTableRanges.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV; 
	descriptorTableRanges.NumDescriptors = 1; 
	descriptorTableRanges.BaseShaderRegister = 0; 
	descriptorTableRanges.RegisterSpace = 0; 
	descriptorTableRanges.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND; 	
	D3D12_ROOT_DESCRIPTOR_TABLE descriptorTable;
	descriptorTable.NumDescriptorRanges = 1; 
	descriptorTable.pDescriptorRanges = &descriptorTableRanges; 
	ZeroMemory(&parameters[0], sizeof(parameters[0]));
	parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;
	parameters[0].ParameterType =  D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
	parameters[0].DescriptorTable = descriptorTable;	
	ZeroMemory(&parameters[1], sizeof(parameters[1]));
	parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
	parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
	D3D12_STATIC_SAMPLER_DESC samplers;
	ZeroMemory(&samplers, sizeof(samplers));
	samplers.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
	samplers.AddressU = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	samplers.AddressV = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	samplers.AddressW = D3D12_TEXTURE_ADDRESS_MODE_BORDER;
	D3D12_ROOT_SIGNATURE_DESC descRootSignature = {2, parameters, 1, &samplers, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
	ID3DBlob* rootBlob;
	D3D12SerializeRootSignature(&descRootSignature, D3D_ROOT_SIGNATURE_VERSION_1, &rootBlob, 0);
	mDevice->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(), IID_PPV_ARGS(&mRootSignature));
	static const D3D12_INPUT_ELEMENT_DESC layout[] =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0,
		D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12,
		D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
	};
	ID3DBlob* vertexShader;
	D3DCompile(Shaders, sizeof(Shaders),"", 0, 0,"VSMain", "vs_5_0", 0, 0, &vertexShader, 0);
	ID3DBlob* pixelShader;
	D3DCompile(Shaders, sizeof(Shaders),"", 0, 0,"PSMain", "ps_5_0", 0, 0, &pixelShader, 0);
	D3D12_RASTERIZER_DESC rasterizer ={D3D12_FILL_MODE_SOLID,D3D12_CULL_MODE_NONE,0,D3D12_DEFAULT_DEPTH_BIAS,D3D12_DEFAULT_DEPTH_BIAS_CLAMP,0.0f,1,0,0,0,(D3D12_CONSERVATIVE_RASTERIZATION_MODE)0};		
	D3D12_BLEND_DESC blendstate = { 0, 0,{0, 0, (D3D12_BLEND)1, (D3D12_BLEND)0, D3D12_BLEND_OP_ADD, (D3D12_BLEND)1, (D3D12_BLEND)0, D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, D3D12_COLOR_WRITE_ENABLE_ALL} };
	D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc ;
	ZeroMemory(&psoDesc, sizeof(psoDesc));
	psoDesc.VS.BytecodeLength = vertexShader->GetBufferSize();
	psoDesc.VS.pShaderBytecode = vertexShader->GetBufferPointer();
	psoDesc.PS.BytecodeLength = pixelShader->GetBufferSize();
	psoDesc.PS.pShaderBytecode = pixelShader->GetBufferPointer();
	psoDesc.pRootSignature = mRootSignature;
	psoDesc.NumRenderTargets = 1;
	psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
	psoDesc.DSVFormat = DXGI_FORMAT_UNKNOWN;
	psoDesc.InputLayout.NumElements = _countof(layout);
	psoDesc.InputLayout.pInputElementDescs = layout;
	psoDesc.RasterizerState = rasterizer;
	psoDesc.BlendState = blendstate;
	psoDesc.SampleDesc.Count = 1;
	psoDesc.DepthStencilState.DepthEnable = false;
	psoDesc.DepthStencilState.StencilEnable = false;
	psoDesc.SampleMask = 0xFFFFFFFF;
	psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
	mDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPSO));
	D3D12_VERTEX_BUFFER_VIEW VBV;
	D3D12_INDEX_BUFFER_VIEW IBV;
	static const float cb[16] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
	for (int i = 0; i < QUEUE_SLOT_COUNT; ++i) 
	{
		static D3D12_HEAP_PROPERTIES uploadHeapProperties = {D3D12_HEAP_TYPE_UPLOAD,D3D12_CPU_PAGE_PROPERTY_UNKNOWN,D3D12_MEMORY_POOL_UNKNOWN,1,1};
		static D3D12_RESOURCE_DESC constantBufferDesc ={D3D12_RESOURCE_DIMENSION_BUFFER,0,sizeof(cb),1,1,1,DXGI_FORMAT_UNKNOWN,{1,0},( D3D12_TEXTURE_LAYOUT)1,(D3D12_RESOURCE_FLAGS)0};
		mDevice->CreateCommittedResource(&uploadHeapProperties,D3D12_HEAP_FLAG_NONE,&constantBufferDesc,D3D12_RESOURCE_STATE_GENERIC_READ,0,IID_PPV_ARGS(&mConstantBuffers[i]));
		void* p;
		mConstantBuffers[i]->Map(0, 0, &p);
		memcpy(p, &cb, sizeof(cb));
		mConstantBuffers[i]->Unmap(0, 0);		
		struct Vertex{float position[3];float uv[2];};
		static const Vertex vertices[4] = {{ { -1.0f, 1.0f, 0 },{ 0, 0 } },{ { 1.0f, 1.0f, 0 },{ 1, 0 } },{ { 1.0f, -1.0f, 0 },{ 1, 1 } },{ { -1.0f, -1.0f, 0 },{ 0, 1 } }};
		static const int indices[6] = {0, 1, 2, 2, 3, 0};
		static const int uploadBufferSize = sizeof(vertices) + sizeof(indices);
		static D3D12_HEAP_PROPERTIES uploadHeapProperties1 ={D3D12_HEAP_TYPE_UPLOAD,D3D12_CPU_PAGE_PROPERTY_UNKNOWN,D3D12_MEMORY_POOL_UNKNOWN,1,1};		
		static D3D12_RESOURCE_DESC uploadBufferDesc ={D3D12_RESOURCE_DIMENSION_BUFFER,0,uploadBufferSize,1,1,1,DXGI_FORMAT_UNKNOWN,{1,0},( D3D12_TEXTURE_LAYOUT)1,(D3D12_RESOURCE_FLAGS)0};
		mDevice->CreateCommittedResource(&uploadHeapProperties1,D3D12_HEAP_FLAG_NONE,&uploadBufferDesc,D3D12_RESOURCE_STATE_GENERIC_READ,0,IID_PPV_ARGS(&mUploadBuffer));
		static D3D12_HEAP_PROPERTIES defaultHeapProperties ={D3D12_HEAP_TYPE_DEFAULT,D3D12_CPU_PAGE_PROPERTY_UNKNOWN,D3D12_MEMORY_POOL_UNKNOWN,1,1};			
		static D3D12_RESOURCE_DESC vertexBufferDesc ={D3D12_RESOURCE_DIMENSION_BUFFER,0,sizeof(vertices),1,1,1,DXGI_FORMAT_UNKNOWN,{1,0},( D3D12_TEXTURE_LAYOUT)1,(D3D12_RESOURCE_FLAGS)0};		
		mDevice->CreateCommittedResource(&defaultHeapProperties,D3D12_HEAP_FLAG_NONE,&vertexBufferDesc,D3D12_RESOURCE_STATE_COPY_DEST,0,IID_PPV_ARGS(&mVertexBuffer));
		static D3D12_RESOURCE_DESC indexBufferDesc ={D3D12_RESOURCE_DIMENSION_BUFFER,0,sizeof(indices),1,1,1,DXGI_FORMAT_UNKNOWN,{1,0},( D3D12_TEXTURE_LAYOUT)1,(D3D12_RESOURCE_FLAGS)0};			
		mDevice->CreateCommittedResource(&defaultHeapProperties,D3D12_HEAP_FLAG_NONE,&indexBufferDesc,D3D12_RESOURCE_STATE_COPY_DEST,0,IID_PPV_ARGS(&mIndexBuffer));
		VBV.BufferLocation = mVertexBuffer->GetGPUVirtualAddress();
		VBV.SizeInBytes = sizeof(vertices);
		VBV.StrideInBytes = sizeof(Vertex);
		IBV.BufferLocation = mIndexBuffer->GetGPUVirtualAddress();
		IBV.SizeInBytes = sizeof(indices);
		IBV.Format = DXGI_FORMAT_R32_UINT;
		void* pp;
		mUploadBuffer->Map(0, 0, &pp);
		memcpy(pp, vertices, sizeof(vertices));
		memcpy((unsigned char*)(pp) + sizeof(vertices),indices, sizeof(indices));
		mUploadBuffer->Unmap(0, 0);
		uploadCommandList->CopyBufferRegion(mVertexBuffer, 0,mUploadBuffer, 0, sizeof(vertices));
		uploadCommandList->CopyBufferRegion(mIndexBuffer, 0,mUploadBuffer, sizeof(vertices), sizeof(indices));	
		D3D12_RESOURCE_BARRIER barrier[2];
		barrier[0].Transition.pResource = mVertexBuffer;
		barrier[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier[0].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier[0].Transition.StateAfter = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER;
		barrier[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		barrier[1].Transition.pResource = mIndexBuffer;
		barrier[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier[1].Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier[1].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
		barrier[1].Transition.StateAfter = D3D12_RESOURCE_STATE_INDEX_BUFFER;
		barrier[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;				
		uploadCommandList->ResourceBarrier(2, barrier);
		LoadTexture("plasma.ppm",uploadCommandList, mDevice, mImage, mUploadImage, msrvDescriptorHeap);
	}			
	uploadCommandList->Close();
	ID3D12CommandList* commandLists[] = { uploadCommandList };
	mCommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);
	mCommandQueue->Signal(uploadFence, 1);
	HANDLE waitEvent = CreateEvent(0, FALSE, FALSE, 0);
	WaitForFence(uploadFence, 1, waitEvent);
	uploadCommandAllocator->Reset();
	CloseHandle(waitEvent);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		MouseLook();
		Mul(CameraRotationYMatrix,CameraRotationXMatrix,CameraRotationMatrix);
		Mul(CameraTranslationMatrix,CameraRotationMatrix,ViewMatrix);
		Inverse(ViewMatrix,ViewMatrix);
		Mul(ProjectionMatrix,ViewMatrix,MVP);	
		float MVPT[4][4] = 
		{
			MVP[0][0], MVP[1][0], MVP[2][0], MVP[3][0],
			MVP[0][1], MVP[1][1], MVP[2][1], MVP[3][1],
			MVP[0][2], MVP[1][2], MVP[2][2], MVP[3][2],
			MVP[0][3], MVP[1][3], MVP[2][3], MVP[3][3]
		};
		KeyboardMovement();		
		WaitForFence(mFrameFences[mCurrentBackBuffer],mFenceValues[mCurrentBackBuffer], frameFenceEvents[mCurrentBackBuffer]);
		mCommandAllocator[mCurrentBackBuffer]->Reset();
		ID3D12GraphicsCommandList* commandList = mCommandList[mCurrentBackBuffer];
		commandList->Reset(mCommandAllocator[mCurrentBackBuffer], 0);
		D3D12_CPU_DESCRIPTOR_HANDLE renderTargetHandle;
		renderTargetHandle.ptr = mrtDescriptorHeap->GetCPUDescriptorHandleForHeapStart().ptr + mCurrentBackBuffer * mrtvDescriptorIncrSize;
		commandList->OMSetRenderTargets(1, &renderTargetHandle, true, 0);
		commandList->RSSetViewports(1, &mViewport);
		commandList->RSSetScissorRects(1, &mRectScissor);
		D3D12_RESOURCE_BARRIER barrier;
		barrier.Transition.pResource = renderTargets[mCurrentBackBuffer];
		barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
		barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
		barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		commandList->ResourceBarrier(1, &barrier);
		static const float clearColor[] = {0.0f, 0.0f, 1.0f, 1.0f};
		commandList->ClearRenderTargetView(renderTargetHandle,clearColor, 0, 0);		
		mCommandList[mCurrentBackBuffer]->SetPipelineState(mPSO);
		mCommandList[mCurrentBackBuffer]->SetGraphicsRootSignature(mRootSignature);
		void* ppData;
		mConstantBuffers[mCurrentBackBuffer]->Map(0, 0, &ppData);
		memcpy(ppData, MVPT, sizeof(MVPT));
		mConstantBuffers[mCurrentBackBuffer]->Unmap(0, 0);
		ID3D12DescriptorHeap* heaps[] = { msrvDescriptorHeap };
		mCommandList[mCurrentBackBuffer]->SetDescriptorHeaps(1, heaps);
		mCommandList[mCurrentBackBuffer]->SetGraphicsRootDescriptorTable(0,msrvDescriptorHeap->GetGPUDescriptorHandleForHeapStart());
		mCommandList[mCurrentBackBuffer]->SetGraphicsRootConstantBufferView(1,mConstantBuffers[mCurrentBackBuffer]->GetGPUVirtualAddress());
		mCommandList[mCurrentBackBuffer]->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
		mCommandList[mCurrentBackBuffer]->IASetVertexBuffers(0, 1, &VBV);
		mCommandList[mCurrentBackBuffer]->IASetIndexBuffer(&IBV);
		mCommandList[mCurrentBackBuffer]->DrawIndexedInstanced(6, 1, 0, 0, 0);					
		D3D12_RESOURCE_BARRIER barrierRT;
		barrierRT.Transition.pResource = renderTargets[mCurrentBackBuffer];
		barrierRT.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
		barrierRT.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
		barrierRT.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
		barrierRT.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
		barrierRT.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
		ID3D12GraphicsCommandList* cmdList = mCommandList[mCurrentBackBuffer];
		cmdList->ResourceBarrier(1, &barrierRT);
		cmdList->Close();
		ID3D12CommandList* cmdLists[] = { cmdList };
		mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);
		mSwapChain->Present(1, 0);
		mCommandQueue->Signal(mFrameFences[mCurrentBackBuffer], mCurrentFenceValue);
		mFenceValues[mCurrentBackBuffer] = mCurrentFenceValue;
		++mCurrentFenceValue;
		mCurrentBackBuffer = (mCurrentBackBuffer + 1) % QUEUE_SLOT_COUNT;
	}
	for (int i = 0; i < QUEUE_SLOT_COUNT; ++i) 
	{
		WaitForFence(mFrameFences[i], mFenceValues[i], frameFenceEvents[i]);
	}
	for (HANDLE event : frameFenceEvents) CloseHandle(event);
	return 0;
}