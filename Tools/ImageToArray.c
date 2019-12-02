#include <stdio.h>
#include <stdlib.h>

void ImageToArray (char *ppm, char *header, char *name)
{
	FILE *fileWrite = fopen(header, "w");  
	int width, height; 
	char buffer[128]; 
	FILE *fileRead = fopen(ppm, "rb");
	fgets(buffer, sizeof(buffer), fileRead);
	do fgets(buffer, sizeof (buffer), fileRead); while (buffer[0] == '#');
	sscanf (buffer, "%d %d", &width, &height);
	do fgets (buffer, sizeof (buffer), fileRead); while (buffer[0] == '#');
	int size = width * height * 4 * sizeof(unsigned char);
	unsigned char *Texels  = (unsigned char*)malloc(size);
	fprintf(fileWrite, "unsigned char %s[] = {", name);
	for (int i = 0; i < size; i++) 
	{
		Texels[i] = ((i % 4) < 3 ) ? (unsigned char) fgetc(fileRead) : (unsigned char) 255 ;
		fprintf(fileWrite, "%d,", Texels[i]);
		if ( (i+1) % 50 == 0) fprintf(fileWrite, "\n", Texels[i]);
	}
	fprintf(fileWrite, "};");
	printf("Image resolution: %d x %d", width, height);	
	free(Texels);
	fclose(fileRead);
	fclose(fileWrite);	
}

int main(int argc, char **argv) 
{
	ImageToArray ("heightmap.ppm", "heightmap.h", "heightmap");
	return 0;
}