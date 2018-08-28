#include <windows.h>
#include <GL/gl.h>
#include <stddef.h>
#include <math.h>

#define deg2rad(x) (x * 3.14159265358979323846f / 180.0f)
#define ModelPositionX 0.0f
#define ModelPositionY 0.0f
#define ModelPositionZ 0.0f
#define ModelRotationX 0.0f
#define ModelRotationY 0.0f
#define ModelRotationZ 0.0f
#define ModelScaleX 1.0f
#define ModelScaleY 1.0f
#define ModelScaleZ 1.0f
#define CameraPositionX 0.0f
#define CameraPositionY 0.0f
#define CameraPositionZ -5.0f
#define CameraRotationX 0.0f
#define CameraRotationY 0.0f
#define CameraRotationZ 0.0f
#define CameraScaleX 1.0f
#define CameraScaleY 1.0f
#define CameraScaleZ 1.0f
#define ScreenWidth 1920.0f
#define ScreenHeight 1080.0f
#define FieldOfView 60.0f
#define NearClip 0.01f
#define FarClip 1000.0f

typedef GLuint(APIENTRY* PFNGLCREATEPROGRAMPROC)();
typedef GLuint(APIENTRY* PFNGLCREATESHADERPROC)(GLenum t);
typedef void(APIENTRY* PFNGLSHADERSOURCEPROC)(GLuint s, GLsizei c, const char*const*string, const GLint* i);
typedef void(APIENTRY* PFNGLCOMPILESHADERPROC)(GLuint s);
typedef void(APIENTRY* PFNGLATTACHSHADERPROC)(GLuint p, GLuint s);
typedef void(APIENTRY* PFNGLLINKPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLUSEPROGRAMPROC)(GLuint p);
typedef void(APIENTRY* PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *b);
typedef void(APIENTRY* PFNGLBINDBUFFERPROC) (GLenum t, GLuint b);
typedef void(APIENTRY* PFNGLBUFFERDATAPROC) (GLenum t, ptrdiff_t s, const GLvoid *d, GLenum u);
typedef void(APIENTRY* PFNGLBINDVERTEXARRAYPROC) (GLuint a);
typedef void(APIENTRY* PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef void(APIENTRY* PFNGLVERTEXATTRIBPOINTERPROC) (GLuint i, GLint s, GLenum t, GLboolean n, GLsizei k, const void *p);
typedef void(APIENTRY* PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint i);
typedef int(APIENTRY* PFNWGLSWAPINTERVALEXTPROC) (int i);
typedef int(APIENTRY* PFNGLGETUNIFORMLOCATIONPROC) (GLuint p, const char *n);
typedef void(APIENTRY* PFNGLUNIFORM1FVPROC) (GLint k, GLsizei c, const GLfloat *v);
typedef int(APIENTRY* PFNGLGETATTRIBLOCATIONPROC) (GLuint p, const char *n);
typedef void(APIENTRY* PFNGLGENVERTEXARRAYSPROC) (GLsizei n, GLuint *a);
typedef void(APIENTRY* PFNGLUNIFORMMATRIX4FVPROC) (GLint l, GLsizei c, GLboolean t, const GLfloat *v);

static const GLfloat vertices[] = {-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
static const GLfloat colors[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
unsigned int VertexBuffer, ColorBuffer, VertexArrayID;
float offsetX = 0.0f, offsetZ = 0.0f;

static const char* VertexShader = \
	"#version 430 core\n"
	"layout (location=0) in vec3 vertexPosition;"
	"layout (location=1) in vec3 vertexColor;"
	"out vec3 fragmentColor;"	
	"uniform mat4 MVP;"
	"void main()"
	"{"	
		"fragmentColor = vertexColor;"
		"gl_Position = MVP * vec4(vertexPosition,1.0);"
	"}";
	
static const char* FragmentShader = \
	"#version 430 core\n"
	"layout (location=0) out vec4 color;"
	"layout (location=1) in vec3 fragmentColor;"
	"void main()"
	"{"	
		"color = vec4(fragmentColor,1);"
	"}";
	
int MakeShader(const char* VS, const char* FS)
{
	int p = ((PFNGLCREATEPROGRAMPROC)wglGetProcAddress("glCreateProgram"))();
	int s1 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B31);	
	int s2 = ((PFNGLCREATESHADERPROC)wglGetProcAddress("glCreateShader"))(0x8B30);	
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s1,1,&VS,0);
	((PFNGLSHADERSOURCEPROC)wglGetProcAddress("glShaderSource"))(s2,1,&FS,0);	
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s1);
	((PFNGLCOMPILESHADERPROC)wglGetProcAddress("glCompileShader"))(s2);	
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s1);
	((PFNGLATTACHSHADERPROC)wglGetProcAddress("glAttachShader"))(p,s2);	
	((PFNGLLINKPROGRAMPROC)wglGetProcAddress("glLinkProgram"))(p);
	return p;
}

void mul(float mat1[][4], float mat2[][4], float res[][4])
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

float ModelTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,ModelPositionX ,
	0.0f,1.0f,0.0f,ModelPositionY,
	0.0f,0.0f,1.0f,ModelPositionZ,
	0.0f,0.0f,0.0f,1.0f
};

float ModelRotationYMatrix[4][4] = 
{
	cos(deg2rad(ModelRotationY)), 0.0f, sin(deg2rad(ModelRotationY)),0.0f,
	0.0f,1.0f,0.0f,0.0f,
	(-1.0f)*sin(deg2rad(ModelRotationY)),0.0f,cos(deg2rad(ModelRotationY)),0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ModelRotationXMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,cos(deg2rad(ModelRotationX)),(-1.0f)*sin(deg2rad(ModelRotationX)),0.0f,
	0.0f,sin(deg2rad(ModelRotationX)),cos(deg2rad(ModelRotationX)),0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ModelRotationZMatrix[4][4] = 
{
	cos(deg2rad(ModelRotationZ)),(-1.0f)*sin(deg2rad(ModelRotationZ)),0.0f,0.0f,
	sin(deg2rad(ModelRotationZ)),cos(deg2rad(ModelRotationZ)),0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ModelScaleMatrix[4][4] = 
{
	ModelScaleX,0.0f,0.0f,0.0f,
	0.0f,ModelScaleY,0.0f,0.0f,
	0.0f,0.0f,ModelScaleZ,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraTranslationMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,CameraPositionX,
	0.0f,1.0f,0.0f,CameraPositionY,
	0.0f,0.0f,1.0f,CameraPositionZ,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationYMatrix[4][4] = 
{
	cos(deg2rad(CameraRotationY)), 0.0f, sin(deg2rad(CameraRotationY)),0.0f,
	0.0f,1.0f,0.0f,0.0f,
	(-1.0f)*sin(deg2rad(CameraRotationY)),0.0f,cos(deg2rad(CameraRotationY)),0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationXMatrix[4][4] = 
{
	1.0f,0.0f,0.0f,0.0f,
	0.0f,cos(deg2rad(CameraRotationX)),(-1.0f)*sin(deg2rad(CameraRotationX)),0.0f,
	0.0f,sin(deg2rad(CameraRotationX)),cos(deg2rad(CameraRotationX)),0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraRotationZMatrix[4][4] = 
{
	cos(deg2rad(CameraRotationZ)),(-1.0f)*sin(deg2rad(CameraRotationZ)),0.0f,0.0f,
	sin(deg2rad(CameraRotationZ)),cos(deg2rad(CameraRotationZ)),0.0f,0.0f,
	0.0f,0.0f,1.0f,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float CameraScaleMatrix[4][4] = 
{
	CameraScaleX,0.0f,0.0f,0.0f,
	0.0f,CameraScaleY,0.0f,0.0f,
	0.0f,0.0f,(-1.0f)*CameraScaleZ,0.0f,
	0.0f,0.0f,0.0f,1.0f
};

float ProjectionMatrix[4][4] = 
{
	((1.0f/tan(deg2rad(FieldOfView/2.0f)))/(ScreenWidth/ScreenHeight)),0.0f,0.0f,0.0f,
	0.0f,(1.0f/tan(deg2rad(FieldOfView/2.0f))),0.0f,0.0f,
	0.0f,0.0f,(-1.0f)* (FarClip+NearClip)/(FarClip-NearClip),(-1.0f)*(2.0f*FarClip*NearClip)/(FarClip-NearClip),
	0.0f,0.0f,-1.0f,0.0f
};

void inverse( float input[4][4], float k[4][4])
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
	inv[0] = m[5]*m[10]*m[15]-m[5]*m[11]*m[14]-m[9]*m[6]*m[15]+ m[9]*m[7]*m[14]+m[13]*m[6]*m[11]-m[13]*m[7]*m[10];
	inv[4] = -m[4]*m[10]*m[15]+m[4]*m[11]*m[14]+m[8]*m[6]*m[15]-m[8]*m[7]*m[14]-m[12]*m[6]*m[11]+m[12]*m[7]*m[10];
	inv[8] = m[4]*m[9]*m[15]-m[4]*m[11]*m[13]-m[8]*m[5]*m[15]+m[8]*m[7]*m[13]+m[12]*m[5]*m[11]-m[12]*m[7]*m[9];
	inv[12] = -m[4]*m[9]*m[14]+m[4]*m[10]*m[13]+m[8]*m[5]*m[14]-m[8]*m[6]*m[13]-m[12]*m[5]*m[10]+m[12]*m[6]*m[9];
	inv[1] = -m[1]*m[10]*m[15]+m[1]*m[11]*m[14]+m[9]*m[2]*m[15]-m[9]*m[3]*m[14]-m[13]*m[2]*m[11]+m[13]*m[3]*m[10];
	inv[5] = m[0]*m[10]*m[15]-m[0]*m[11]*m[14]-m[8]*m[2]*m[15]+m[8]*m[3]*m[14]+m[12]*m[2]*m[11]-m[12]*m[3]*m[10];
	inv[9] = -m[0]*m[9]*m[15]+m[0]*m[11]*m[13]+m[8]*m[1]*m[15]-m[8]*m[3]*m[13]-m[12]*m[1]*m[11]+m[12]*m[3]*m[9];
	inv[13] = m[0]*m[9]*m[14]-m[0]*m[10]*m[13]-m[8]*m[1]*m[14]+m[8]*m[2]*m[13]+m[12]*m[1]*m[10]-m[12]*m[2]*m[9];
	inv[2] = m[1]*m[6]*m[15]-m[1]*m[7]*m[14]-m[5]*m[2]*m[15]+m[5]*m[3]*m[14]+m[13]*m[2]*m[7]-m[13]*m[3]*m[6];
	inv[6] = -m[0]*m[6]*m[15]+m[0]*m[7]*m[14]+m[4]*m[2]*m[15]-m[4]*m[3]*m[14]-m[12]*m[2]*m[7]+m[12]*m[3]*m[6];
	inv[10] = m[0]*m[5]*m[15]-m[0]*m[7]*m[13]-m[4]*m[1]*m[15]+m[4]*m[3]*m[13]+m[12]*m[1]*m[7]-m[12]*m[3]*m[5];
	inv[14] = -m[0]*m[5]*m[14]+m[0]*m[6]*m[13]+m[4]*m[1]*m[14]-m[4]*m[2]*m[13]-m[12]*m[1]*m[6]+m[12]*m[2]*m[5];
	inv[3] = -m[1]*m[6]*m[11]+m[1]*m[7]*m[10]+m[5]*m[2]*m[11]-m[5]*m[3]*m[10]-m[9]*m[2]*m[7]+m[9]*m[3]*m[6];
	inv[7] = m[0]*m[6]*m[11]-m[0]*m[7]*m[10]-m[4]*m[2]*m[11]+m[4]*m[3]*m[10]+m[8]*m[2]*m[7]-m[8]*m[3]*m[6];
	inv[11] = -m[0]*m[5]*m[11]+m[0]*m[7]*m[9]+m[4]*m[1]*m[11]-m[4]*m[3]*m[9]-m[8]*m[1]*m[7]+m[8]*m[3]*m[5];
	inv[15] = m[0]*m[5]*m[10]-m[0]*m[6]*m[9]-m[4]*m[1]*m[10]+m[4]*m[2]*m[9]+m[8]*m[1]*m[6]-m[8]*m[2]*m[5];
	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	det = 1.0 / det;
	for (i = 0; i < 16; i++) invOut[i] = inv[i] * det;	
	k[0][0] = invOut[0];  k[0][1] = invOut[1];  k[0][2] = invOut[2];  k[0][3] = invOut[3];
	k[1][0] = invOut[4];  k[1][1] = invOut[5];  k[1][2] = invOut[6];  k[1][3] = invOut[7];
	k[2][0] = invOut[8];  k[2][1] = invOut[9];  k[2][2] = invOut[10]; k[2][3] = invOut[11];
	k[3][0] = invOut[12]; k[3][1] = invOut[13]; k[3][2] = invOut[14]; k[3][3] = invOut[15];  
}

void movement()
{
	if (GetAsyncKeyState(0x57)) CameraTranslationMatrix[2][3] += 0.001f;
	if (GetAsyncKeyState(0x53)) CameraTranslationMatrix[2][3] -= 0.001f;
	if (GetAsyncKeyState(0x44)) CameraTranslationMatrix[0][3] += 0.001f;
	if (GetAsyncKeyState(0x41)) CameraTranslationMatrix[0][3] -= 0.001f;	
	if (GetAsyncKeyState(0x45)) CameraTranslationMatrix[1][3] += 0.001f;
	if (GetAsyncKeyState(0x51)) CameraTranslationMatrix[1][3] -= 0.001f;		
	POINT point;
	int mx = (int)ScreenWidth  >> 1;
	int my = (int)ScreenHeight >> 1;
	GetCursorPos(&point);
	if( (point.x == mx) && (point.y == my) ) return;
	SetCursorPos(mx, my);	
	float deltaZ = (float)((mx - point.x)) ;
	float deltaX = (float)((my - point.y)) ;
	if (deltaX>0.0f) offsetX-=0.5f; 
	if (deltaX<0.0f) offsetX+=0.5f; 
	if (deltaZ>0.0f) offsetZ-=0.5f; 
	if (deltaZ<0.0f) offsetZ+=0.5f; 
	CameraRotationXMatrix[1][1] = cos(deg2rad(CameraRotationX+offsetX));
	CameraRotationXMatrix[1][2] = (-1.0f)*sin(deg2rad(CameraRotationX+offsetX));
	CameraRotationXMatrix[2][1] = sin(deg2rad(CameraRotationX+offsetX));
	CameraRotationXMatrix[2][2] = cos(deg2rad(CameraRotationX+offsetX));				
	CameraRotationYMatrix[0][0] = cos(deg2rad(CameraRotationY+offsetZ));
	CameraRotationYMatrix[0][2] = sin(deg2rad(CameraRotationY+offsetZ));
	CameraRotationYMatrix[2][0] = (-1.0f)*sin(deg2rad(CameraRotationY+offsetZ));
	CameraRotationYMatrix[2][2] = cos(deg2rad(CameraRotationY+offsetZ));
}

int main()
{
	PIXELFORMATDESCRIPTOR pfd = { 0, 0, PFD_DOUBLEBUFFER };
	HDC hdc = GetDC(CreateWindow("static", 0, WS_POPUP | WS_VISIBLE | WS_MAXIMIZE, 0, 0, 0, 0, 0, 0, 0, 0));
	SetPixelFormat(hdc, ChoosePixelFormat(hdc, &pfd), &pfd);
	wglMakeCurrent(hdc, wglCreateContext(hdc));
	ShowCursor(0);
	((PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT")) (0);
	((PFNGLGENVERTEXARRAYSPROC)wglGetProcAddress("glGenVertexArrays")) (1, &VertexArrayID);		
	((PFNGLBINDVERTEXARRAYPROC)wglGetProcAddress("glBindVertexArray")) (VertexArrayID);	
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &VertexBuffer);
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(vertices), vertices, 0x88E4);
	((PFNGLGENBUFFERSPROC)wglGetProcAddress("glGenBuffers"))(1, &ColorBuffer);	
	((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, ColorBuffer);
	((PFNGLBUFFERDATAPROC)wglGetProcAddress("glBufferData"))(0x8892, sizeof(colors), colors, 0x88E4);		
	int PS = MakeShader(VertexShader,FragmentShader);	
	((PFNGLUSEPROGRAMPROC)wglGetProcAddress("glUseProgram"))(PS);	
	int MatrixID = ((PFNGLGETUNIFORMLOCATIONPROC)wglGetProcAddress("glGetUniformLocation"))(PS,"MVP"); 	
	do
	{
		movement();
		float ModelRotYX[4][4], ModelRotYXZ[4][4], ModelTR[4][4]; 
		float ModelMatrix[4][4], CameraRotYX[4][4], CameraRotYXZ[4][4]; 
		float CameraTR[4][4], CameraMatrix[4][4], ViewMatrix[4][4];
		float ProjectionViewMatrix[4][4], ModelViewMatrix[4][4], MVP[4][4] ;
		mul(ModelRotationYMatrix,ModelRotationXMatrix,ModelRotYX);
		mul(ModelRotYX,ModelRotationZMatrix,ModelRotYXZ);
		mul(ModelTranslationMatrix,ModelRotYXZ,ModelTR);
		mul(ModelTR,ModelScaleMatrix,ModelMatrix);	
		mul(CameraRotationYMatrix,CameraRotationXMatrix,CameraRotYX);
		mul(CameraRotYX,CameraRotationZMatrix,CameraRotYXZ);
		mul(CameraTranslationMatrix,CameraRotYXZ,CameraTR);
		mul(CameraTR,CameraScaleMatrix,CameraMatrix);
		inverse(CameraMatrix,ViewMatrix);	
		mul(ProjectionMatrix,ViewMatrix,ProjectionViewMatrix);
		mul(ProjectionViewMatrix,ModelMatrix,MVP);	
		mul(ViewMatrix,ModelMatrix,ModelViewMatrix);
		float MVPT[4][4] = 
		{
			MVP[0][0], MVP[1][0],MVP[2][0],MVP[3][0],
			MVP[0][1], MVP[1][1],MVP[2][1],MVP[3][1],
			MVP[0][2],MVP[1][2], MVP[2][2],MVP[3][2],
			MVP[0][3],MVP[1][3], MVP[2][3], MVP[3][3]
		};
		glClear(GL_COLOR_BUFFER_BIT);
		((PFNGLUNIFORMMATRIX4FVPROC)wglGetProcAddress("glUniformMatrix4fv"))(MatrixID, 1, GL_FALSE, &MVPT[0][0]);
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(0);
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, VertexBuffer);
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(0,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );		
		((PFNGLENABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glEnableVertexAttribArray"))(1);	
		((PFNGLBINDBUFFERPROC)wglGetProcAddress("glBindBuffer"))(0x8892, ColorBuffer);		
		((PFNGLVERTEXATTRIBPOINTERPROC)wglGetProcAddress("glVertexAttribPointer"))(1,3, GL_FLOAT, GL_FALSE, 0,(void*)0 );		
		glDrawArrays(GL_TRIANGLES, 0, 3);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(0);
		((PFNGLDISABLEVERTEXATTRIBARRAYPROC)wglGetProcAddress("glDisableVertexAttribArray"))(1);		
		wglSwapLayerBuffers(hdc, WGL_SWAP_MAIN_PLANE);
	} while (!GetAsyncKeyState(VK_ESCAPE));
	return 0;
}