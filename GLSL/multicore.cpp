// cl multicore.cpp opengl32.lib user32.lib gdi32.lib  /EHsc
#include <windows.h>
#include <GL/gl.h>
#include <vector>
#include <math.h>
#include <thread>

const int width = 1280;
const int height = 720;
float buffer[width][height][3] = {};
const int NUM_THREADS = 12; 

void mainImage(std::vector<int> rows) 
{
	for( int i = 0; i < rows.size(); ++i ) 
	{
		for( int x = 0; x < width; ++x ) 
		{
			int y = rows[i];
			float iTime = GetTickCount()*0.001f;
			float dx = (2.0f*(float)x-(float)width)/(float)height;
			float dy = (2.0f*(float)y-(float)height)/(float)height;
			float dz = 2.0f;
			float length = sqrtf(dx*dx+dy*dy+dz*dz);
			dx = dx / length;
			dy = dy / length;
			dz = dz / length;
			float px = dx*(-1.0f/dy);
			float py = dy*(-1.0f/dy);
			float pz = dz*(-1.0f/dy);
			for(int j=1; j<24; j++)
			{ 
				px += 0.5f*(sinf(iTime + 0.5f*pz * float(j)));
				pz += 0.5f*(cosf(iTime + 0.5f*px * float(j)));
			}
			buffer[x][y][0] = fmin(0.5f+0.5f*cosf(4.0f+px),1.0f) * -dy * 3.0f;
			buffer[x][y][1] = fmin(0.5f+0.5f*cosf(4.0f+pz+2.0f),1.0f) * -dy * 3.0f;
			buffer[x][y][2] = fmin(0.5f+0.5f*cosf(4.0f+px+4.0f),1.0f) * -dy * 3.0f;
		}
	}
}

void GenerateData() 
{
	std::thread th[NUM_THREADS];
	int N = 0, rows = height / NUM_THREADS;
	std::vector<std::vector<int>> buckets( NUM_THREADS ); 
	for (int y = 0; y < height; ++y) 
	{
		buckets[N].push_back(y);
		if ( (buckets[N].size() == rows) && (N < (NUM_THREADS - 1)) ) {N++;}
	}
	for( int i = 0; i < NUM_THREADS; ++i ) 
	{
		th[i] = std::thread(mainImage, buckets[i]);
	}
	for( int i = 0; i < NUM_THREADS; ++i ) 
	{
		th[i].join();
	}
} 

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
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "Demo"};
	RegisterClass(&win);
	HDC hdc = GetDC(CreateWindowEx(0, win.lpszClassName, "Demo", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height, 0, 0, 0, 0));
	PIXELFORMATDESCRIPTOR pfd = { 0,0,PFD_DOUBLEBUFFER };
	SetPixelFormat(hdc,ChoosePixelFormat(hdc,&pfd),&pfd);
	wglMakeCurrent(hdc,wglCreateContext(hdc));
	glOrtho(0.0, width, 0.0, height, -1.0, 1.0);
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		GenerateData();
		glBegin( GL_POINTS );
		for (int y = 0; y < height; ++y) 
		{
			for (int x = 0; x < width; ++x) 
			{
				glVertex2i( x, y );
				glColor3f(buffer[x][y][0], buffer[x][y][1], buffer[x][y][2]);
			}
		}
		glEnd();
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	};
	return 0;
}