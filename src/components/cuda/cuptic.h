/**
 * This is profiler independent interface that is shared by the
 * cupti profiler and event APIs. The functions defined here are
 * common (cuptic_) functions that sit alongside the profiler
 * functions, such as cuptip_ctx_open and cuptie_ctx_open.
 */
#ifndef __CUPTIC_H__
#define __CUPTIC_H__

#include <stdio.h>
#include "papi.h"
#include "papi_memory.h"
#include "papi_internal.h"
#include "cupti_config.h"

/* CUDA Driver function pointers */
extern CUresult (*cuCtxGetCurrentPtr)(CUcontext *ctx);
extern CUresult (*cuCtxSetCurrentPtr)(CUcontext ctx);
extern CUresult (*cuCtxDestroyPtr)(CUcontext ctx);
extern CUresult (*cuCtxCreatePtr)(CUcontext *ctx, unsigned int flags, CUdevice dev);
extern CUresult (*cuCtxPopCurrentPtr)(CUcontext *ctx);
extern CUresult (*cuCtxPushCurrentPtr)(CUcontext ctx);
extern CUresult (*cuCtxSynchronizePtr)(void);
extern CUresult (*cuCtxGetDevicePtr)(CUdevice *dev);
extern CUresult (*cuDeviceGetPtr)(CUdevice *dev, int id);
extern CUresult (*cuDeviceGetCountPtr)(int *count);
extern CUresult (*cuDeviceGetNamePtr)(char *name, int len, CUdevice dev);
extern CUresult (*cuDeviceGetAttributePtr)(int *val, CUdevice_attribute attr, CUdevice dev);
extern CUresult (*cuDevicePrimaryCtxRetainPtr)(CUcontext *ctx, CUdevice dev);
extern CUresult (*cuDevicePrimaryCtxReleasePtr)(CUdevice dev);
extern CUresult (*cuInitPtr)(unsigned int flags);
extern CUresult (*cuGetErrorStringPtr)(CUresult res, const char **err_string);

/* CUDA runtime function pointers */
extern cudaError_t (*cudaGetDeviceCountPtr)(int *count);
extern cudaError_t (*cudaGetDevicePtr)(int *id);
extern cudaError_t (*cudaSetDevicePtr)(int id);
extern cudaError_t (*cudaGetDevicePropertiesPtr)(struct cudaDeviceProp *prop, int id);
extern cudaError_t (*cudaDeviceGetAttributePtr)(int *value, enum cudaDeviceAttr attr, int id);
extern cudaError_t (*cudaFreePtr)(void *ptr);
extern cudaError_t (*cudaDriverGetVersionPtr)(int *ver);
extern cudaError_t (*cudaRuntimeGetVersionPtr)(int *ver);

/* CUPTI common function pointers */
extern CUptiResult (*cuptiGetVersionPtr)(uint32_t *version);

/* cupti dlopen handler */
extern void *cupti_dlp;
extern char error_string[PAPI_MAX_STR_LEN];
extern unsigned int _cuda_lock;

/* cupti attributes definition */
struct cuptic_info {
    CUcontext ctx;
    CUcontext primary;
};

typedef struct cuptic_info *cuptic_info_t;
typedef int64_t cuptic_bitmap_t;

int cuptic_init(void);
int cuptic_shutdown(void);
int cuptic_info_create(cuptic_info_t *info);
int cuptic_info_update(cuptic_info_t info, const char **events, int num_events);
int cuptic_info_destroy(cuptic_info_t info);
int cuptic_api_fallback_event(void);
int cuptic_api_check_version(void);
int cuptic_dev_get_map(const char **events, int num_events, cuptic_bitmap_t *bitmap);
int cuptic_dev_acquire(cuptic_bitmap_t bitmap);
int cuptic_dev_release(cuptic_bitmap_t bitmap);
int cuptic_dev_get_id(cuptic_bitmap_t bitmap, int dev_count, int *device_id);
int cuptic_dev_get_count(cuptic_bitmap_t bitmap, int *num_devices);
int cuptic_evt_get_device(const char *string, int *device);
int cuptic_err_get_last(const char **err_string);

#endif /* End of __cuptic_H__ */
