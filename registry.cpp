//Basic read and write from Windows Registry. Be careful ! You modify the registry at your own risk !
//Przemyslaw Zaworski, 2017

#include <windows.h>
#include <iostream>

void registry_read (LPCTSTR subkey,LPCTSTR name,DWORD type)
{
	HKEY key;
	TCHAR value[255];
	DWORD value_length = 255;
	RegOpenKey(HKEY_LOCAL_MACHINE,subkey,&key);
	RegQueryValueEx(key,name, NULL, &type, (LPBYTE)&value, &value_length);
	RegCloseKey(key);
	std::cout <<value<<std::endl;
}

void registry_write (LPCTSTR subkey,LPCTSTR name,DWORD type,const char* value)
{
	HKEY key;
	RegOpenKey(HKEY_LOCAL_MACHINE,subkey,&key);
	RegSetValueEx(key, name, 0, type, (LPBYTE)value, strlen(value)*sizeof(char));
	RegCloseKey(key);
}

int main()
{
	registry_read("Hardware\\Description\\System\\CentralProcessor\\0","ProcessorNameString",REG_SZ);
	registry_write("Hardware\\Description\\System\\CentralProcessor\\0","NewValue",REG_SZ,"Content");
	return 0;
}