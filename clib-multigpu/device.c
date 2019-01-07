#include "device.h"

#include "memorymanager.h"

#include "debug.h"
#include "utils.h"

crossbowDeviceP crossbowDeviceCreate (int id) {
	crossbowDeviceP p;
	p = (crossbowDeviceP) crossbowMalloc(sizeof(crossbow_device_t));
	memset (p, 0, sizeof(crossbow_device_t));
	p->id = id;
	return p;
}

void crossbowDeviceSelect (crossbowDeviceP p) {
	nullPointerException(p);
	p->selected = 1;
}

unsigned crossbowDeviceSelected (crossbowDeviceP p) {
	nullPointerException(p);
	return (p->selected == 1);
}

void crossbowDeviceFree (crossbowDeviceP p) {
	if (! p)
		return;
	crossbowFree (p, sizeof(crossbow_device_t));
	return;
}
