// cl fixed.c opengl32.lib user32.lib gdi32.lib
#include <windows.h>
#include <GL/gl.h>
#include <math.h>

#define width 1920
#define height 1080

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

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	ShowCursor(0);
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_POPUP, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glOrtho(0.0, width, 0.0, height, -1.0, 1.0);
	int x = 0;
	int y = 0;
	float *vertices = (float*) malloc(width * height * sizeof(float) * 3);
	float *colors = (float*) malloc(width * height * sizeof(float) * 3);
	for(int i=0; i<(width*height*3); i++)
	{
		float uvx = (2.0f * x - (float)width) / (float)height;
		float uvy = (2.0f * y - (float)height) / (float)height;
		float length = 0.1f / sqrtf(uvx*uvx + uvy*uvy);
		if ((i % 3) == 0) 
		{
			vertices[i] = x;
			colors[i] = length;
		}
		if ((i % 3) == 1) 
		{
			vertices[i] = y;
			colors[i] = length;
		}
		if ((i % 3) == 2) 
		{
			vertices[i] = 0.0f;
			colors[i] = 0.0f;
			if (x == width) x = 0;
			if (x % width == 0) y++;
			x++;
		}
	}	
	glEnableClientState (GL_VERTEX_ARRAY);
	glEnableClientState (GL_COLOR_ARRAY);
	glVertexPointer (3, GL_FLOAT, 0, vertices);
	glColorPointer (3, GL_FLOAT, 0, colors);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		glDrawArrays(GL_POINTS, 0, width * height);
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	};
	return 0;
}