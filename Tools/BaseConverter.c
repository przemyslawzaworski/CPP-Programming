#include <stdio.h>
#include <stdlib.h>

char* BaseConverter(int number, int baseFrom, int baseTo)
{
	int length = 32;
	char* result = (char*)malloc(length * sizeof(char));
	int decimal = 0;
	int power = 1;
	while (number > 0) 
	{
		int digit = number % 10;
		decimal += digit * power;
		power *= baseFrom;
		number /= 10;
	}
	int index = 0;
	do 
	{
		if (index >= length - 1) 
		{
			length *= 2;
			result = (char*)realloc(result, length * sizeof(char));
		}
		int remainder = decimal % baseTo;
		result[index++] = (remainder < 10) ? remainder + '0' : (remainder - 10) + 'A';
		decimal /= baseTo;
	} while (decimal > 0);
	result[index] = '\0';
	int start = 0;
	int end = index - 1;
	while (start < end) 
	{
		char temp = result[start];
		result[start] = result[end];
		result[end] = temp;
		start++;
		end--;
	}
	return result;
}

int main() 
{
	int number, baseFrom, baseTo;
	printf("Enter the number to convert: ");
	scanf("%d", &number);
	printf("Enter the base of the number: ");
	scanf("%d", &baseFrom);
	printf("Enter the base to convert to: ");
	scanf("%d", &baseTo);
	char* output = BaseConverter(number, baseFrom, baseTo);
	printf("Converted number: %s\n", output);
	free(output);
	return 0;
}