%{
	#include "global.h"
%}
%token	PROGRAM
%token	BEGINN
%token	END
%token	VAR
%token	INTEGER
%token	REAL
%token	ARRAY
%token	OF
%token	FUN
%token	PROC
%token	IF
%token	THEN
%token	ELSE
%token	DO
%token	WHILE
%token	RELOP
%token	MULOP
%token	SIGN
%token	ASSIGN
%token	OR
%token	NOT
%token	ID
%token	NUM
%token	NONE
%token	DONE
%%
start:	PROGRAM ID '(' start_params ')' ';' 
	declarations 
	subprogram_declarations		{cout << "Semantic action 1\n";}  
	compound_statement '.' 		{cout << "Semantic action 2\n";} eof
	;
start_params:	ID
	| start_params ',' ID 		
	;			
identifier_list : 	ID	{cout << "Semantic action 3\n";}
	| identifier_list  ',' ID 	{cout << "Semantic action 4\n";}
	;
declarations:	declarations VAR identifier_list  ':' type ';'  
	{ cout << "Semantic action 5\n"; if($5==INTEGER || $5 == REAL) cout << "Semantic action 6\n";}
	| 
	;
type:	standard_type	
	| ARRAY '[' NUM '.' '.' NUM ']' OF standard_type	{ cout << "Semantic action 7\n";}
	;
standard_type:	INTEGER | REAL	
	;
subprogram_declarations: subprogram_declarations subprogram_declaration ';'	
	|
	;					
subprogram_declaration: subprogram_head declarations compound_statement {cout << "Semantic action 8\n";}
	;
subprogram_head:	FUN ID 	{cout << "Semantic action 9\n";} 
	arguments {cout << "Semantic action 10\n";}
	':' standard_type {cout << "Semantic action 11\n";}  
	';'  
	| PROC ID {cout << "Semantic action 12\n";}
	arguments {	cout << "Semantic action 13\n";} ';' 
	;		
arguments: '(' parameter_list ')' {cout << "Semantic action 14\n";}	 
	|
	;
parameter_list:	identifier_list ':' type	{cout << "Semantic action 15\n";}														
	| parameter_list ';' identifier_list  ':' type  {cout << "Semantic action 16\n";}	  
	;
compound_statement:	BEGINN optional_statement END	
	; 
optional_statement: 	statement_list			
	|		 
	;
statement_list: 	statement 						
	| statement_list ';' statement 			
	;
statement:	variable ASSIGN simple_expression {cout << "Semantic action 17\n";}
	| procedure_statement				
	| compound_statement				
	| IF expression 	{cout << "Semantic action 18\n";}
	THEN statement  	{cout << "Semantic action 19\n";}
	ELSE statement		{cout << "Semantic action 20\n";}
	| WHILE 			{cout << "Semantic action 21\n";}  
	expression DO 		{cout << "Semantic action 22\n";}
	statement 			{cout << "Semantic action 23\n";} 
	;
variable:	ID	{cout << "Semantic action 24\n";}
	| ID '[' simple_expression ']'		{cout << "Semantic action 25\n";}	
	;
procedure_statement: 	ID {cout << "Semantic action 26\n";}
	| ID '(' expression_list ')'	{cout << "Semantic action 27\n";}
	;
expression_list:	expression	{cout << "Semantic action 28\n";} 
	| expression_list ',' expression	{cout << "Semantic action 29\n";}
	;
expression:		simple_expression				{cout << "Semantic action 30\n";}
	| simple_expression RELOP simple_expression	{cout << "Semantic action 31\n";}
	;
simple_expression:	term					
	| SIGN term							{cout << "Semantic action 32\n";} 
	| simple_expression SIGN term		{cout << "Semantic action 33\n";}
	| simple_expression OR term		{cout << "Semantic action 34\n";}
	; 
term: factor 
	| term MULOP factor		{cout << "Semantic action 35\n";}
	;
factor:	variable	{cout << "Semantic action 36\n";}								
	| ID '(' expression_list ')' 	{cout << "Semantic action 37\n";}
	| NUM					
	| '(' expression ')'			{cout << "Semantic action 38\n";}
	| NOT factor					{cout << "Semantic action 39\n";}
	;
eof: DONE	{return 0; }
	;
%%
void yyerror(char const *s)
{
  printf("Line %d: %s \n",lineno,s);
}