//function 'conversion' converts natural number between numerals systems (from base 1 to 10)
//for example 'conversion(21013,5,9)' converts number 21013 from quinary to nonary system (should returns 1806)
//code written by Przemyslaw Zaworski

#include <iostream>
#include <math.h>
#include <string> 
using namespace std;

static string conversion (int n, int p, int w)
{
	string number = to_string(n);
	int d = 1;
	int k = 0;
	for (int i=number.length(); i>0; --i)
	{
		int x = number[i-1]-'0';
		k += (x*d);
		d *= p;
	}
	string r="";
	int t,z; 
	while (true)
	{
		r = to_string(k%w)+r;
		k = int(trunc(k/w));
		if (k==0) break;
	}
	return r; 
}

int main()
{
	string output = conversion(21013,5,9);
	cout << output;
}
