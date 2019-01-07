#ifndef __CROSSBOW_SYSTEM_H_
#define __CROSSBOW_SYSTEM_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <float.h>

#include "macros.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "../debug.h"
#include "../utils.h"

#include "../stream.h"

#include "../variable.h"
#include "../variableschema.h"

#include "../model.h"

#include "../databuffer.h"

#include "../memorymanager.h"

#include "../localvariable.h"

#include "../kernelconfigurationparameter.h"

#ifdef __cplusplus
}
#endif

#endif /* __CROSSBOW_SYSTEM_H_ */
