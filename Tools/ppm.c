// cl.exe ppm.c user32.lib gdi32.lib shell32.lib
#include <windows.h>
#include <stdio.h>
#include <shellapi.h>

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

int WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
	AllocConsole();
	freopen("CONOUT$", "w+", stdout);
	int argc;
	char** argv;
	LPWSTR* lpArgv = CommandLineToArgvW( GetCommandLineW(), &argc );
	argv = (char**)malloc( argc*sizeof(char*) );
	int s = 0;
	for( int i=0; i < argc; ++i )
	{
		s = wcslen( lpArgv[i] ) + 1;
		argv[i] = (char*)malloc( s );
		wcstombs( argv[i], lpArgv[i], s );
	}
	if (argc != 2)
	{
		ExitProcess(0);
	}
	LocalFree(lpArgv);
	ShowWindow(GetConsoleWindow(), SW_HIDE);	
	FreeConsole();	
	
	int width, height; 
	char buffer[128]; 
	FILE *file = fopen(argv[1], "rb");
	fgets(buffer, sizeof(buffer), file);
	do fgets(buffer, sizeof (buffer), file); while (buffer[0] == '#');
	sscanf (buffer, "%d %d", &width, &height);
	do fgets (buffer, sizeof (buffer), file); while (buffer[0] == '#');
	int size = width * height * 4 * sizeof(unsigned char);
	unsigned char *pixels  = (unsigned char *)malloc(size);
	int counter = 0;
	for (int i = 0; i < (size/4); i++)
	{
		unsigned char r = (unsigned char) fgetc(file);
		unsigned char g = (unsigned char) fgetc(file);
		unsigned char b = (unsigned char) fgetc(file);
		unsigned char a = (unsigned char) 255;
		pixels[counter] = b;
		pixels[counter+1] = g;
		pixels[counter+2] = r;
		pixels[counter+3] = a;
		counter += 4;
	}
	fclose(file);
	
	int exit = 0;
	MSG msg;
	WNDCLASS win = {CS_OWNDC|CS_HREDRAW|CS_VREDRAW, WindowProc, 0, 0, 0, 0, 0, (HBRUSH)(COLOR_WINDOW+1), 0, "PPM Viewer"};
	RegisterClass(&win);
	HWND hwnd = CreateWindowEx(0, win.lpszClassName, "PPM Viewer", WS_VISIBLE|WS_OVERLAPPEDWINDOW, 0, 0, width, height+40, 0, 0, 0, 0);
	HDC hdc = GetDC(hwnd);
	BITMAPINFO bmi = {{sizeof(BITMAPINFOHEADER),width,height,1,32,BI_RGB,0,0,0,0,0},{0,0,0,0}};	
	while (!exit)
	{
		while(PeekMessage(&msg, 0, 0, 0, PM_REMOVE))
		{
			if( msg.message==WM_QUIT ) exit = 1;
			TranslateMessage( &msg );
			DispatchMessage( &msg );
		}
		if (hwnd == GetActiveWindow()) 
			StretchDIBits(hdc,0,height,width,-height,0,0,width,height,pixels,&bmi,DIB_RGB_COLORS,SRCCOPY);
	}
	free(pixels);
	free(argv);
	return 0;
}