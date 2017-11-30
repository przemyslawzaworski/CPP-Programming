SOURCE=main.c parser.c lexer.c
OBJECTS=$(SOURCE:.cpp=.o)

.cpp.o:
	g++ -c -o  $@ $<

compiler.exe: $(OBJECTS)
	g++ -o $@ $^

${OBJECTS}: global.h

parser.c parser.h: parser.y
	bison --defines=parser.h -o parser.c parser.y

lexer.c: lexer.l
	flex -t lexer.l > lexer.c