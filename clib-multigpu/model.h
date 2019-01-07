#ifndef __CROSSBOW_MODEL_H_
#define __CROSSBOW_MODEL_H_

#include "variable.h"
#include "databuffer.h"

#include "solverconfiguration.h"

#include "debug.h"

#include <pthread.h>

/*
 * Op[1] --> Op[2] --> ... --> Op[N]
 *   |         |                 |
 *  Var       Var               null
 *   |         |
 *  Var       null
 *   |
 *  null
 */
typedef struct crossbow_model *crossbowModelP;
typedef struct crossbow_model {
	int id;
	int ops;
	int vars;
	
	int dev;

	int updates;
	volatile int clock;

	int offset;
	int elements;
	int bytes;
    
    /* Flag to distinguish between base models and replicas */
    int base;

	crossbowDataBufferP data;
	crossbowDataBufferP gradient; /* Gradient per model */

	crossbowDataBufferP last;
	crossbowDataBufferP diff;
	crossbowDataBufferP temp;
#ifdef ADAGRAD
	crossbowDataBufferP hist;
	crossbowDataBufferP tmp1;
#endif
	int wpc;
	crossbowModelUpdate_t type;
	crossbowSolverConfP conf;

	crossbowVariableP *variables;
	/*
	 * Model lock, to help during synchronisation
	 *
	 * The lock is acquired by the GPU worker thread before a dataflow is scheduled.
	 *
	 * We could have locking at at finer granularity, like the read/write locks  of
	 * the CPU models. For example, when a dataflow operator is scheduled, it could
	 * acquire a read or a write lock based on its requirements.
	 *
	 * However, the execution of dataflow operators is asynchronous with respect to
	 * the GPU worker thread.
	 *
	 * A solution would be to acquire the (read or write) lock, record a CUDA event
	 * and then pass the event to an event handler thread pool. Upon completion  of
	 * the operator's GPU functions the event would fire and then it would be  safe
	 * to unlock the model.
	 */
	pthread_mutex_t lock;
	/*
	 * More locks: GPU events to synchronise with the parameter server (realised as another stream).
	 */
#ifdef UPDATE_MODEL_INCREMENTALLY
	cudaEvent_t *client;
	cudaEvent_t *server;
#else
	cudaEvent_t client;
	cudaEvent_t server;
#endif

	cudaEvent_t updated;
	cudaEvent_t accumulated;

} crossbow_model_t;

crossbowModelP crossbowModelCreate (int, int, int, int);

void crossbowModelFinalise (crossbowModelP);

crossbowModelP crossbowModelReplicate (crossbowModelP, int, int);

void crossbowModelRegister (crossbowModelP, int, int, crossbowVariableSchemaP);

int crossbowModelVariableCount (crossbowModelP, int);

crossbowVariableP crossbowModelFind (crossbowModelP, int, int);

crossbowDataBufferP crossbowModelVariable (crossbowModelP, int, int, int *, int *);

crossbowDataBufferP crossbowModelVariableAccelerated (crossbowModelP, int, int, int *, int *, cublasHandle_t);

crossbowDataBufferP crossbowModelVariableGradient (crossbowModelP, int, int, int *, int *);

crossbowDataBufferP crossbowModelVariableLastGradient (crossbowModelP, int, int, int *, int *);

float crossbowModelGetLearningRateForVariable (crossbowModelP, int, int, int);

int crossbowModelClock (crossbowModelP);

void crossbowModelLock (crossbowModelP);

int crossbowModelTryLock (crossbowModelP);

void crossbowModelUnlock (crossbowModelP);

void crossbowModelIncrementClock (crossbowModelP, int);

void crossbowModelStore (crossbowModelP, const char *);

void crossbowModelLoad  (crossbowModelP, const char *);

void crossbowModelFree (crossbowModelP);

void crossbowModelDump (crossbowModelP p);

#endif /* __CROSSBOW_MODEL_H_ */
