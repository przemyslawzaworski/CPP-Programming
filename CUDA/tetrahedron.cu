/*
	Code written by Przemyslaw Zaworski https://github.com/przemyslawzaworski
	References: https://github.com/straaljager/GPU-path-tracing-tutorial-2 , http://www.iquilezles.org/www/articles/derivative/derivative.htm
	Compile from scratch (set *.inc and *lib files manually in new project) or for simplicity, 
	replace code with simpleGL.cu (http://docs.nvidia.com/cuda/cuda-samples/index.html#simple-opengl)
	Visual Studio: built in "Release" mode (x64).
*/

#include <helper_math.h> 
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define width 1280   //screen width
#define height 720   //screen height

float t = 0.0f;   //timer
float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer
  
__device__  float plane(float3 p, float3 c, float3 n)   //plane signed distance field
{
	return dot(p - c, n);
}

__device__  float tetrahedron(float3 p, float e)   //tetrahedron signed distance field, created from planes intersection
{
	float f = 0.57735;
	float a = plane(p, make_float3(e, e, e), make_float3(-f, f, f));
	float b = plane(p, make_float3(e, -e, -e), make_float3(f, -f, f));
	float c = plane(p, make_float3(-e, e, -e), make_float3(f, f, -f));
	float d = plane(p, make_float3(-e, -e, e), make_float3(-f, -f, -f));
	return max(max(a, b), max(c, d));
}

__device__ float map(float3 p,float t)   //virtual geometry
{
	p = make_float3(p.x, cos(t)*p.y + sin(t)*p.z, -sin(t)*p.y + cos(t)*p.z);
	p = make_float3(cos(t)*p.x - sin(t)*p.z, p.y, sin(t)*p.x + cos(t)*p.z);
	return tetrahedron(p, 2.0f);
}

__device__ float3 lighting(float3 p, float t, float e)   //directional derivative based lighting
{
	float3 a = make_float3(0.1f, 0.1f, 0.1f);   //ambient light color
	float3 ld = normalize(make_float3(6.0f, 15.0f, -7.0f));   //light direction
	float3 lc = make_float3(0.95f, 0.95f, 0.0f);   //light color
	float s = (map(p + ld*e, t) - map(p, t)) / e;   
	float3 d = clamp(s, 0.0, 1.0)*lc + a;
	return clamp(d,0.0f,1.0f);   //final color
}

__device__ float3 raymarch(float3 ro, float3 rd, float k)   //raymarching
{
	for (int i = 0; i<128; i++)
	{
		float p = map(ro,k);
		if (p < 0.01) return lighting(ro,k,0.01f);
		ro += p*rd;
	}
	return make_float3(0.0f, 0.0f, 0.0f);   //background color
}

__global__ void rendering(float3 *output,float k)
{   																												
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1)*width + x;
	float2 resolution = make_float2((float)width, (float)height);   //screen resolution
	float2 coordinates = make_float2((float)x, (float)y);   //fragment coordinates
	float2 uv = (2.0*coordinates - resolution) / resolution.y;
	float3 ro = make_float3(0.0f, 0.0f, -8.0f);   //ray origin
	float3 rd = normalize(make_float3(uv, 2.0f));   //ray direction
	float3 c = raymarch(ro, rd, k);
	float colour;
	unsigned char bytes[] = {(unsigned char)(c.x*255+0.5), (unsigned char)(c.y*255+0.5), (unsigned char)(c.z*255+0.5), 1};
	memcpy(&colour, &bytes, sizeof(colour));   //convert from 4 bytes to single float
	output[i] = make_float3(x, y, colour);
}

void time(int x) 
{
	if (glutGetWindow() )
	{
		glutPostRedisplay();
		glutTimerFunc(10, time, 0);
		t+=0.0166f;
	}
} 

void display(void)
{
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	rendering <<< grid, block >>>(device,t);   //execute kernel
	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}

int main(int argc, char** argv) 
{
	cudaMalloc(&device, width * height * sizeof(float3));   //allocate memory on the GPU VRAM
	glutInit(&argc, argv);   //OpenGL initializing
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(width, height);
	glutCreateWindow("Basic CUDA OpenGL raymarching - tetrahedron");
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	glutDisplayFunc(display);
	time(0);
	glewInit();
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	unsigned int size = width * height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
	glutMainLoop();   //event processing loop
	cudaFree(device);
}