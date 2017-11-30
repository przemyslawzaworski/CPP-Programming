#include "global.h"
#include "parser.h"

int main(int argc, char* argv[]) 
{	
	FILE *input_file = fopen(argv[1],"r");	
	yyin = input_file;
	int result = yyparse();
	fclose(input_file);
	return 0;
}

