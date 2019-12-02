#include <stdio.h>
#include <stdlib.h>
 
float* LoadMapFromFile (const char* filename, int size)
{
	FILE* binFile = fopen(filename,"rb");
	float* source = (float*) malloc(size*sizeof(float));
	fread(source, sizeof(float), size, binFile);
	fclose(binFile);
	return source;
}

void MapToArray (char *bin, char *header, char *name, int trianglecount)
{
	FILE *fileWrite = fopen(header, "w");  
	float* source = LoadMapFromFile ("uvmap.bin", trianglecount * 2);
	fprintf(fileWrite, "float %s[] = {", name);	
	for (int i = 0; i < (trianglecount * 2); i++) 
	{  
		fprintf(fileWrite, "%.3f,", source[i]);
		if ( (i+1) % 20 == 0) fprintf(fileWrite, "\n", source[i]);
	}
	fprintf(fileWrite, "};");
	free (source);
	fclose(fileWrite);
}

int main(int argc, char **argv) 
{
	MapToArray("uvmap.bin", "uv.h", "uvarray", 1572867);
	return 0;
}