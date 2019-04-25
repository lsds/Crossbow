#ifndef __CROSSBOW_UTILS_H_
#define __CROSSBOW_UTILS_H_

#undef __INPUT_ISPINNED_
/* #define __INPUT_ISPINNED_ */

/* #undef __LAZY_MATERIALISATION */
#define __LAZY_MATERIALISATION

#undef __LAZY_MAPPING
/* #define __LAZY_MAPPING */

/* #define __NVPROF */
/* #define __NVPROF_MARK_TASK */

/* #define __NVPROF_MARK_OPERATOR */

#define USE_CUDNN
/* #undef USE_CUDNN */

#define USE_TASKHANDLERS
/* #undef USE_TASKHANDLERS */

/* #define SHARD_AXPY */
#undef SHARD_AXPY

/* #define UPDATE_MODEL_INCREMENTALLY */
#undef UPDATE_MODEL_INCREMENTALLY

#define INTRA_TASK_MEASUREMENTS
/* #undef INTRA_TASK_MEASUREMENTS */

/* #define INTER_TASK_MEASUREMENTS */
#undef INTER_TASK_MEASUREMENTS

#define INTER_TASK_MEASUREMENTS_DISPLAY_INTERVAL 1

/* #define MAKESPAN_MEASUREMENTS */
#undef MAKESPAN_MEASUREMENTS

/* #define REPRODUCIBILITY */
#undef REPRODUCIBILITY

/* #define ADAGRAD */
#undef ADAGRAD

/* #define TRAIN_WITH_MASTER */
#undef TRAIN_WITH_MASTER

#define CHECK_WITH_MASTER
/* #undef CHECK_WITH_MASTER */

#define DEVICE_SYNCHRONIZE
// #undef DEVICE_SYNCHRONIZE

// #define ELASTIC_AVERAGE
#undef ELASTIC_AVERAGE

/* #define EAMSGD__NORMALIZE */
#undef EAMSGD__NORMALIZE

#define EAMSGD__APPLY_MOMENTUM
// #undef EAMSGD__APPLYMOMENTUM

// #define EAMSGD__SHARE_MOMENTUM
#undef EAMSGD__SHARE_MOMENTUM

/* #define USE_NCCL */
#undef USE_NCCL

/* #define MAP_RECORDS */
#undef MAP_RECORDS

#define max(a,b) ((a) > (b) ? (a) : (b))
#define min(a,b) ((a) < (b) ? (a) : (b))

typedef void (*crossbowKernelFunctionP)(void *);

typedef enum crossbow_databuffer_type { PIN = 0, REF } crossbowDataBuffer_t;

typedef enum crossbow_phase { TRAIN = 0, CHECK } crossbowPhase_t;

typedef enum crossbow_local_variable_type { RO = 0, RW } crossbowLocalVariable_t;

typedef enum crossbow_kernel_configuration_parameter_type { INT = 0, FLOAT, INT_ARRAY, FLOAT_ARRAY, DOUBLE, UNDEFINED } crossbowKernelConfigParam_t;

typedef enum crossbow_kernel_scalar_type { I32 = 0, F32, F64, UND } crossbowKernelScalar_t;

typedef enum crossbow_cudnn_kernel_type { NONE = 0, CONV, POOL, RELU, SOFTMAX, BATCHNORM, DROPOUT } crossbowCudnnKernel_t;

typedef enum crossbow_model_synchronisation_type { BSP = 0, SSP, ASP } crossbowModelSynchronisation_t;

typedef enum crossbow_model_update_type { DEFAULT = 0, WORKER, EAMSGD, SYNCHRONOUSEAMSGD, DOWNPOUR } crossbowModelUpdate_t;

typedef enum crossbow_learning_rate_decay_policy_type { FIXED = 0, INV, STEP, MULTISTEP, LSR, EXP, CLR } crossbowLearningRateDecayPolicy_t;

typedef enum crossbow_operator_dependency_type { START_BEFORE_START = 0, END_BEFORE_START } crossbowOperatorDependency_t;

typedef enum crossbow_model_synchronisation_mode_type { SINGLE_GPU = 0, MULTI_GPU } crossbowModelSynchronisationMode_t;

typedef enum crossbow_momentum_method_type { POLYAK = 0, NESTEROV } crossbowMomentumMethod_t;

typedef enum crossbow_lightweight_dataset_operation_type { RESERVE = 0, RELEASE } crossbowLightWeightDatasetOp_t;

typedef enum crossbow_list_node_item_type { INTPTR = 0, VOIDPTR } crossbowListNodeItem_t;

typedef enum crossbow_image_data_format_type { HWC = 0, CHW } crossbowImageDataFormat_t;

#endif /* __CROSSBOW_UTILS_H_ */
