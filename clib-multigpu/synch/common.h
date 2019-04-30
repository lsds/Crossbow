#ifndef __CROSSBOW_SYNCHRONISATION_COMMON_H_
#define __CROSSBOW_SYNCHRONISATION_COMMON_H_

#include "../executioncontext.h"

void crossbowSynchronisationAccumulateGradientsAcrossDevices (crossbowExecutionContextP ctx, crossbowModelP defaultModel, crossbowDeviceP defaultDev);

void crossbowSynchronisationSynchroniseModelAcrossDevices (crossbowExecutionContextP ctx, crossbowModelP defaultModel, crossbowDeviceP defaultDev, unsigned shareMomentum);

void crossbowSynchronisationSynchroniseModelOnDevice (crossbowExecutionContextP ctx, int first, crossbowModelP model, crossbowDeviceP dev);

#endif /* __CROSSBOW_SYNCHRONISATION_COMMON_H_ */