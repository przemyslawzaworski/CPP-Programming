/* Output:
4.000000
11.000000
-123.500000
----------------------------------------------
11.000000
-123.500000
*/

#include <stdio.h>
#include <stdlib.h>

struct List
{
  size_t size;
  float* elements;
};

void Add(struct List* list, float value)
{
  list->size++;
  list->elements = realloc(list->elements, list->size * sizeof(float));
  list->elements[list->size - 1] = value; 
}

void RemoveAt(struct List* list, size_t index)
{
  if (index >= list->size) return;
  for (int i=index; i<list->size; i++) list->elements[i] = list->elements[i+1];
  list->size--;
  list->elements = realloc(list->elements, list->size * sizeof(float));
}

int main(void) 
{
  struct List list = {0, NULL};
  Add(&list, 4.0f);
  Add(&list, 11.0f);
  Add(&list, -123.5f);
  for (int i=0; i<list.size; i++) printf("%f\n",list.elements[i]);
  printf("----------------------------------------------\n");
  RemoveAt(&list, 0);
  for (int i=0; i<list.size; i++) printf("%f\n",list.elements[i]);
  return 0;
}
