%option noyywrap
%{
#include "parser.h"
#include "global.h"
int lineno = 1;
%}

letter			[a-zA-Z]
digit				[0-9]
relop				"="|"<>"|"<"|"<="|">="|">"
mulop				"*"|"mod"|"/"|"and"|"div"
function		"function"
procedure		"procedure"
assign			":="
sign				"+"|"-"	
array				"array"
of					"of"
integer			"integer"
realword		"real"
int					{digit}+
exponent		[eE][+-]?{int}
real				{int}("."{int}){exponent}?
id					{letter}({letter}|{digit})*

%%
[\t | " "]+	
[ \n ]			{lineno++;} 
program			{return PROGRAM;}
begin				{return BEGINN;}
end					{return END;}
var					{return VAR;}
{integer}		{yylval = INTEGER; return INTEGER;}
{realword}	{yylval = REAL; return REAL;}
not					{return NOT;}
or					{return OR;}
if					{return IF;}
then				{return THEN;}
else				{return ELSE;}
do					{return DO;}
while				{return WHILE;}
{array}			{return ARRAY;}
{of}				{return OF;}
{procedure}	{return PROC;}
{function}	{return FUN;}
{relop}			{return RELOP;}
{mulop}			{return MULOP;}
{sign}			{return SIGN;}
{assign}		{return ASSIGN;}
{id}				{return ID;}
{int}				{return NUM;}
{real}			{return NUM; }
<<EOF>>			{return DONE;}
.						{return *yytext;}
%%