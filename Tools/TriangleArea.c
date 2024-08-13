#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct 
{
    float* vertices;
    float area;
} Triangle;

float TriangleArea(const float* vertices) 
{
    float abx = vertices[3] - vertices[0];
    float aby = vertices[4] - vertices[1];
    float abz = vertices[5] - vertices[2];
    float acx = vertices[6] - vertices[0];
    float acy = vertices[7] - vertices[1];
    float acz = vertices[8] - vertices[2];
    float crossx = aby * acz - abz * acy;
    float crossy = abz * acx - abx * acz;
    float crossz = abx * acy - aby * acx;
    return sqrt(crossx * crossx + crossy * crossy + crossz * crossz) * 0.5f;
}

void WriteToFile(const char* filename, const Triangle* triangle) 
{
    FILE* file = fopen(filename, "wb");
    fwrite(triangle->vertices, sizeof(float), 9, file);
    fwrite(&triangle->area, sizeof(float), 1, file);
    fclose(file);
}

void ReadFromFile(const char* filename, Triangle* triangle) 
{
    FILE* file = fopen(filename, "rb");
    triangle->vertices = (float*)malloc(9 * sizeof(float));
    fread(triangle->vertices, sizeof(float), 9, file);
    fread(&triangle->area, sizeof(float), 1, file);
    fclose(file);
}

void WriteData()
{
    Triangle triangle;
    triangle.vertices = (float*)malloc(9 * sizeof(float));
    float vertices[9] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    memcpy(triangle.vertices, vertices, 9 * sizeof(float));   
    triangle.area = TriangleArea(triangle.vertices);
    printf("The area of the triangle is: %f\n", triangle.area);   
    WriteToFile("triangle.dat", &triangle);
    free(triangle.vertices);
}

void ReadData()
{
    Triangle triangle;
    ReadFromFile("triangle.dat", &triangle);
    printf("The area of the triangle is: %f\n", triangle.area);
    free(triangle.vertices);
}

int main() 
{
    WriteData();
    ReadData();
    return 0;
}
