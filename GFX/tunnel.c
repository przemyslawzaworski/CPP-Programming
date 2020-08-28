// Compile: cl tunnel.c -O2
// Multithreading software rendering demo
// To fix: screen tearing
#include <windows.h>
#include <math.h>
#include <stdio.h>
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")

#define ScreenWidth 1280
#define ScreenHeight 720
#define NumThreads 12

typedef struct {float x; float y;} Vector2;
typedef struct {float x; float y; float z;} Vector3;
typedef struct {int threadId; float threadTime;} ThreadInfo;

static int RenderTarget[ScreenWidth*ScreenHeight];

int ColorToInt (Vector3 c)
{
	return 255 << 24 | (unsigned char)(c.x * 255) << 16 | (unsigned char)(c.y * 255) << 8 | (unsigned char)(c.z * 255);
}

LPCSTR IntToString (int number)
{
	char* chars = malloc(16);
	sprintf(chars, "%d", number);
	strcat(chars, " FPS");  // Additional line of code
	LPCSTR string = chars;
	free(chars);
	return string;
}

float Min(float a, float b)
{
	return a < b ? a : b;
}

float Max(float a, float b)
{
	return a > b ? a : b;
}

float Clamp(float x, float a, float b)
{
	return Max(a, Min(b, x));
}

float Smoothstep(float a, float b, float x)
{
	float t = Clamp((x - a)/(b - a), 0.0f, 1.0f);
	return t * t * (3.0f - (2.0f * t));
}

float Mod(float x, float y)
{
	return x - y * floorf(x/y);
}

Vector3 Normalize(Vector3 p)
{
	float q = sqrtf(p.x * p.x + p.y * p.y + p.z * p.z); 
	Vector3 vector = {p.x / q, p.y / q, p.z / q};
	return vector;
}

float Length(Vector3 p)
{
	return sqrtf(p.x * p.x + p.y * p.y + p.z * p.z);
}

float Grid(float px, float py)
{
	return 0.4 + Smoothstep(0.94f, 1.0f, Max(Mod(3.0f*px, 1.03f), Mod(3.0f*py, 1.03f)));
}

void MainImage (int fragCoordX, int fragCoordY, float iTime)
{
	float ux = ((float)fragCoordX - ScreenWidth * 0.5f) / ScreenHeight + 0.5f * cosf(iTime*0.5f);
	float uy = ((float)fragCoordY - ScreenHeight * 0.5f) / ScreenHeight + 0.25f * sinf(iTime*0.5f);
	Vector3 uv = {ux, uy, 1.0f};
	Vector3 rd = Normalize(uv);
	float sdf = 1.0f / powf(powf(fabs(rd.x)*0.75f, 6.0f) + powf(fabs(rd.y), 6.0f), 1.0f/6.0f);
	Vector3 surface = {rd.x * sdf, rd.y * sdf, iTime + rd.z * sdf};
	Vector3 light = {0.0f - surface.x, 0.0f - surface.y, iTime + 3.0f - surface.z};
	float atten = 1.0f / (0.75f + Length(light) * 0.5f);
	float f = Clamp((Grid(surface.y,surface.z) + Grid(surface.x,surface.z)) * atten * atten * atten, 0.0f, 1.0f);
	Vector3 fragColor = {f, f, f};
	RenderTarget[fragCoordY*ScreenWidth+fragCoordX] = ColorToInt(fragColor);
}

DWORD WorkerThread (ThreadInfo* info)
{
	int x = 0;
	int y = (info->threadId * ScreenHeight / NumThreads);
	int start = y * ScreenWidth;
	int end = ScreenWidth * ((info->threadId + 1) * ScreenHeight / NumThreads);
	for (int i = start; i < end ; i++)
	{
		MainImage(x, y, info->threadTime);
		x++;
		if (x == ScreenWidth) {x = 0; y++;}
	}
	return 0;
}

LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	if (uMsg==WM_CLOSE || uMsg==WM_DESTROY || (uMsg==WM_KEYDOWN && wParam==VK_ESCAPE))
	{
		PostQuitMessage(0); return 0;
	}
	else
	{
		return DefWindowProc(hWnd, uMsg, wParam, lParam);
	}
}

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	int exit = 0;
	MSG msg;
	WNDCLASS wnd = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&wnd);
	HWND hwnd = CreateWindowEx(0, wnd.lpszClassName, "Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, ScreenWidth, ScreenHeight, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	BITMAPINFO bmi = {{sizeof(BITMAPINFOHEADER), ScreenWidth, ScreenHeight, 1, 32, BI_RGB, 0, 0, 0, 0, 0}, {0, 0, 0, 0}};
	ThreadInfo info[NumThreads];
	HANDLE* threads = (HANDLE*)malloc(sizeof(HANDLE) * NumThreads);
	int frame = 0;
	float lastTime = 0.0f;
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float time = (float)GetTickCount() * 0.001f;
		for (int i = 0; i < NumThreads; i++)
		{
			info[i].threadId = i;
			info[i].threadTime = time;
			threads[i] = CreateThread(0, 0, WorkerThread, &info[i], 0, 0);
		}
		int result = WaitForMultipleObjects(NumThreads, threads, 0, INFINITE);
		for (int i = 0; i < NumThreads; i++) CloseHandle(threads[i]);
		StretchDIBits(hdc, 0, 0, ScreenWidth, ScreenHeight, 0, 0, ScreenWidth, ScreenHeight, RenderTarget, &bmi, DIB_RGB_COLORS, SRCCOPY);
		frame++;
		if (time - lastTime > 1.0f)
		{
			lastTime = time;
			SetWindowTextA(hwnd, IntToString(frame));
			frame = 0;
		}
	}
	free(threads);
	return 0;
}
