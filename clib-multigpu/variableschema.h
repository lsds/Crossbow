#ifndef __CROSSBOW_VARIABLESCHEMA_H_
#define __CROSSBOW_VARIABLESCHEMA_H_

typedef struct crossbow_variable_schema *crossbowVariableSchemaP;
typedef struct crossbow_variable_schema {
	int dims;
	int *shape;
	int elements;
	int bytes;
} crossbow_variable_schema_t;

crossbowVariableSchemaP crossbowVariableSchemaCreate (int, int *, int);

crossbowVariableSchemaP crossbowVariableSchemaCopy (crossbowVariableSchemaP);

int crossbowVariableSchemaEqual (crossbowVariableSchemaP, crossbowVariableSchemaP);

int crossbowVariableSchemaCountElementsInRange (crossbowVariableSchemaP, int, int);

int crossbowVariableSchemaCountElementsFrom (crossbowVariableSchemaP, int);

int crossbowVariableSchemaShape (crossbowVariableSchemaP, int);

char *crossbowVariableSchemaString (crossbowVariableSchemaP);

void crossbowVariableSchemaFree (crossbowVariableSchemaP);

#endif /* __CROSSBOW_VARIABLESCHEMA_H_ */
