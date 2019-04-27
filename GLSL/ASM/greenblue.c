// gcc greenblue.c -lX11 -lGL -o greenblue
// ./greenblue
// Tested in Lubuntu 19

#include <X11/X.h>
#include <sys/time.h>
#define GL_GLEXT_PROTOTYPES 1
#include <GL/gl.h>
#include <GL/glx.h>
#include <string.h>

static const char* FS = \
	"!!ARBfp1.0\n"
	"PARAM time = program.env[0];"
	"TEMP R0, R1, R2, H0;"
	"SIN R1.x, time.x;"
	"MAD R2.x, R1.x, {0.5}.x, {0.5}.x;"
	"MUL R0.x, fragment.position, {0.00078};"
	"SGE H0.x, R2.x, R0;"
	"MAD result.color, H0.x, {0,.5,-1,0}, {0,0,1,1};"
	"END";

static long GetTickCount( void )
{
	struct timeval now, res;
	gettimeofday(&now, 0);
	return (long)((now.tv_sec*1000) + (now.tv_usec/1000));
}

int main( void )
{
	Display *hDisplay = XOpenDisplay( NULL );
	int buffer[]  = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_DOUBLEBUFFER, None };
	XVisualInfo *info = glXChooseVisual( hDisplay, DefaultScreen(hDisplay), buffer);
	XSetWindowAttributes winAttr;
	winAttr.override_redirect = 1;
	winAttr.colormap = XCreateColormap( hDisplay, RootWindow(hDisplay, info->screen), info->visual, AllocNone );
	Window hWnd = XCreateWindow( hDisplay, RootWindow(hDisplay, info->screen), 0, 0, 1280, 720, 0, info->depth, InputOutput, info->visual, CWColormap|CWOverrideRedirect, &winAttr);
	XMapRaised(hDisplay, hWnd);
	XGrabKeyboard(hDisplay, hWnd, 1, GrabModeAsync, GrabModeAsync, CurrentTime);
	glXMakeCurrent( hDisplay, hWnd, glXCreateContext(hDisplay, info, NULL, 1) );
	glEnable(GL_FRAGMENT_PROGRAM_ARB);
	glProgramStringARB (GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, strlen(FS), FS);
	XEvent event;
	long s = GetTickCount();
	while( !XCheckTypedEvent(hDisplay, KeyPress, &event) )
	{    
		GLfloat t = (GetTickCount()-s) * 0.001f;
		glProgramEnvParameter4fvARB(GL_FRAGMENT_PROGRAM_ARB, 0, &t);   
		glRects( -1, -1, 1, 1 );
		glXSwapBuffers( hDisplay, hWnd );
	}
	return 0;
}
