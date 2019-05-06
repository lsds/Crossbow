#ifndef __CROSSBOW_SYNCHRONISATION_COMMON_H_
#define __CROSSBOW_SYNCHRONISATION_COMMON_H_

#include "../executioncontext.h"

void crossbowSynchronisationAllReduceGradientsAcrossDevices (crossbowExecutionContextP);

void crossbowSynchronisationAccumulateGradientsAcrossDevices (crossbowExecutionContextP, crossbowModelP, crossbowDeviceP);

void crossbowSynchronisationSynchroniseModelAcrossDevices (crossbowExecutionContextP, crossbowModelP, crossbowDeviceP, unsigned);

void crossbowSynchronisationSynchroniseModelOnDevice (crossbowExecutionContextP, int, crossbowModelP, crossbowDeviceP);

#endif /* __CROSSBOW_SYNCHRONISATION_COMMON_H_ */
