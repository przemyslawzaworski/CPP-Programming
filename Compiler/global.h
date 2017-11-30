#include <stdio.h>
#include <iostream>
using namespace std;
extern int lineno;
int yylex();
int yyparse();
void yyerror(char const* s);
extern FILE* yyin;