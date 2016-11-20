#include <iostream>
#include "stdio.h"

using namespace std;

void show();
int main(void){
	char a='0';
	fprintf(stdout, "%s\n", "Hello world!");
	cin >> a;
	cout<< a << endl;
	cout << "Hello world!\n";
	show();
	return 0;
}
void show(){
	printf("%s\n", "Hello world!");
}
