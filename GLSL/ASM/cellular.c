// Compile with Visual Studio: cl cellular.c opengl32.lib user32.lib gdi32.lib

#include <windows.h>

typedef void (__stdcall* PFNGLPROGRAMSTRINGARBPROC) (enum target, enum format, int len, const void *string);
typedef void (__stdcall* PFNGLPROGRAMENVPARAMETER4FVARBPROC) (enum target, int index, const float *params);
 
static const char* FS = \
	"!!ARBfp1.0\n"
	"PARAM time = program.env[0];"
	"TEMP R0, R1, R2;"
	"MOV R0.x, {0.4};"
	"MUL R0.z, R0.x, time.x;"
	"MUL R0.xy, fragment.position, {0.005}.x;"
	"DP3 R2.z, R0, {1.2, 0.8, 0.4};"
	"DP3 R2.x, R0, {-0.4, 0.8, -0.8};"
	"DP3 R2.y, R0, {-0.8, 0.4, 1.2};"
	"DP3 R1.z, R2, {-0.72, -0.48, -0.24};"
	"DP3 R1.x, R2, {0.24, -0.48, 0.48};"
	"DP3 R1.y, R2, {0.48, -0.24, -0.72};"
	"DP3 R0.z, R1, {1.152, 0.768, 0.384};"
	"DP3 R0.x, R1, {-0.384, 0.768, -0.768};"
	"DP3 R0.y, R1, {-0.768, 0.384, 1.152};"
	"FRC R0.xyz, R0;"
	"ADD R0.xyz, R0, {-0.5}.x;"
	"DP3 R0.x, R0, R0;"
	"RSQ R0.x, R0.x;"
	"RCP R0.w, R0.x;"
	"FRC R0.xyz, R1;"
	"ADD R0.xyz, R0, {-0.5}.x;"
	"DP3 R0.x, R0, R0;"
	"FRC R1.xyz, R2;"
	"ADD R1.xyz, R1, {-0.5}.x;"
	"DP3 R0.y, R1, R1;"
	"RSQ R0.x, R0.x;"
	"RSQ R0.y, R0.y;"
	"RCP R0.x, R0.x;"
	"RCP R0.y, R0.y;"
	"MIN R0.x, R0.y, R0;"
	"MIN R0.x, R0, R0.w;"
	"MUL result.color, R0.x, {0.86805552, 1.3020833, 2.6041665}.xyzx;"
	"END";

static LRESULT CALLBACK WindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
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
	ShowCursor(0);
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, " "};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, " ", WS_VISIBLE|WS_POPUP, 0, 0, 1920, 1080, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = {0, 0, PFD_DOUBLEBUFFER};
	SetPixelFormat(hdc, ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	glEnable(0x8804);	
	((PFNGLPROGRAMSTRINGARBPROC)wglGetProcAddress("glProgramStringARB")) (0x8804, 0x8875, strlen(FS), FS);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message == WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		float time = GetTickCount()*0.001f;
		((PFNGLPROGRAMENVPARAMETER4FVARBPROC)wglGetProcAddress("glProgramEnvParameter4fvARB"))(0x8804, 0, &time);
		glRects(-1, -1, 1, 1);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	}
	return 0;
}