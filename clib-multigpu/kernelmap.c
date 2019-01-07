#include "kernelmap.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

static unsigned int crossbowKernelHash (const char *s) {
	unsigned int hash = 5381;
	int c;
	while ((c = *s++))
		hash = ((hash << 5) + hash) + c;
	return hash;
}

crossbowKernelMapP crossbowKernelMapCreate (int slots) {
	int i;
	crossbowKernelMapP map = (crossbowKernelMapP) crossbowMalloc (sizeof(crossbow_kernel_map_t));
	map->slots = slots;
	map->count = 0;
	map->bin = (crossbowKernelBindingP *) crossbowMalloc (map->slots * sizeof(crossbowKernelBindingP));
	for (i = 0; i < map->slots; i++)
		map->bin[i] = NULL;
	return map;
}

void crossbowKernelMapBind (crossbowKernelMapP map, char *name, crossbowKernelFunctionP func) {
	unsigned int h;
	crossbowKernelBindingP p;
	h = crossbowKernelHash (name) % map->slots;
	p = (crossbowKernelBindingP) crossbowMalloc (sizeof(crossbow_kernel_binding_t));
	p->name = crossbowStringCopy (name);
	p->func = func;
	p->next = map->bin[h]; /* Chain node at head of slot */
	map->bin[h] = p;
	map->count ++;
	return;
}

crossbowKernelFunctionP crossbowKernelMapResolve (crossbowKernelMapP map, const char *name) {
	crossbowKernelBindingP p;
	unsigned int h;
	h = crossbowKernelHash (name) % map->slots;

	for (p = map->bin[h]; p != NULL; p = p->next) {
		if (strcmp (name, p->name) == 0)
			return p->func;
	}
	return NULL;
}

void crossbowKernelMapDump (crossbowKernelMapP map) {
	int i;
	int len, max = 0;
	crossbowKernelBindingP p;
	printf ("=== [Kernelmap: %d kernels] ===\n", map->count);
	for (i = 0; i < map->slots; i++) {
		len = 0;
		for (p = map->bin[i]; p != NULL; p = p->next) {
			if (len == 0)
				printf("[%04d]: ", i);
			printf ("%s -> ", p->name);
			len++;
		}
		if (len > 0)
			printf ("null (%d)\n", len);

		if (max < len)
			max = len;
	}
	printf ("=== [End of kernelmap dump] ===\n");
	fflush(stdout);
	return;
}

void crossbowKernelMapFree (crossbowKernelMapP map) {
	int i;
	if (! map)
		return;
	crossbowKernelBindingP p, q;
	for (i = 0; i < map->slots; i++) {
		for (p = map->bin[i]; p != NULL; p = q) {
			q = p->next;
			crossbowStringFree(p->name);
			crossbowFree(p, sizeof(crossbow_kernel_binding_t));
		}
	}
	crossbowFree (map->bin, map->slots * sizeof (crossbowKernelBindingP));
	crossbowFree (map, sizeof (crossbow_kernel_map_t));
	return;
}
