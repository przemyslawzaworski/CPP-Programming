//Windows: g++ -o generate_html.exe generate_html.cpp
#include <iostream>
#include <cstdlib>
#include <fstream>
using namespace std;

int main()
{
	fstream html_file("index.html",std::ios::out);
	string content = "<HTML><HEAD></HEAD><BODY><B>Hello World!</B></BODY></HTML>";
	html_file.write(&content[0],content.length() );	
	html_file.close();
	system("index.html");
	return 0;
}