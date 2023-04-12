#include <dlfcn.h>
#include "papi.h"
#include "papi_memory.h"
#include "cuptic.h"

/* CUDA Driver function pointers */
CUresult (*cuCtxGetCurrentPtr)(CUcontext *ctx);
CUresult (*cuCtxSetCurrentPtr)(CUcontext ctx);
CUresult (*cuCtxDestroyPtr)(CUcontext ctx);
CUresult (*cuCtxCreatePtr)(CUcontext *ctx, unsigned int flags, CUdevice dev);
CUresult (*cuCtxPopCurrentPtr)(CUcontext *ctx);
CUresult (*cuCtxPushCurrentPtr)(CUcontext ctx);
CUresult (*cuCtxSynchronizePtr)(void);
CUresult (*cuCtxAttachPtr)(CUcontext *ctx, unsigned int flags);
CUresult (*cuCtxDetachPtr)(CUcontext ctx);
CUresult (*cuCtxGetDevicePtr)(CUdevice *dev);
CUresult (*cuDeviceGetPtr)(CUdevice *dev, int id);
CUresult (*cuDeviceGetCountPtr)(int *count);
CUresult (*cuDeviceGetNamePtr)(char *name, int len, CUdevice dev);
CUresult (*cuDeviceGetAttributePtr)(int *val, CUdevice_attribute attr, CUdevice dev);
CUresult (*cuDevicePrimaryCtxRetainPtr)(CUcontext *ctx, CUdevice dev);
CUresult (*cuDevicePrimaryCtxReleasePtr)(CUdevice dev);
CUresult (*cuInitPtr)(unsigned int flags);
CUresult (*cuGetErrorStringPtr)(CUresult res, const char **err_string);

/* CUDA runtime function pointers */
cudaError_t (*cudaGetDeviceCountPtr)(int *count);
cudaError_t (*cudaGetDevicePtr)(int *id);
cudaError_t (*cudaSetDevicePtr)(int id);
cudaError_t (*cudaGetDevicePropertiesPtr)(struct cudaDeviceProp *prop, int id);
cudaError_t (*cudaDeviceGetAttributePtr)(int *value, enum cudaDeviceAttr attr, int id);
cudaError_t (*cudaFreePtr)(void *ptr);
cudaError_t (*cudaDriverGetVersionPtr)(int *ver);
cudaError_t (*cudaRuntimeGetVersionPtr)(int *ver);

/* CUPTI common function pointers */
CUptiResult (*cuptiGetVersionPtr)(uint32_t *version);

void *cupti_dlp;
char error_string[PAPI_MAX_STR_LEN];
static void *cuda_dlp;
static void *cu_dlp;
static cuptic_bitmap_t global_device_map;
unsigned int _cuda_lock;

static int
load_cuda_driver_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = "libcuda.so";
    cuda_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (cuda_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    cuCtxSetCurrentPtr           = dlsym(cuda_dlp, "cuCtxSetCurrent");
    cuCtxGetCurrentPtr           = dlsym(cuda_dlp, "cuCtxGetCurrent");
    cuCtxDestroyPtr              = dlsym(cuda_dlp, "cuCtxDestroy");
    cuCtxCreatePtr               = dlsym(cuda_dlp, "cuCtxCreate");
    cuCtxGetDevicePtr            = dlsym(cuda_dlp, "cuCtxGetDevice");
    cuDeviceGetPtr               = dlsym(cuda_dlp, "cuDeviceGet");
    cuDeviceGetCountPtr          = dlsym(cuda_dlp, "cuDeviceGetCount");
    cuDeviceGetNamePtr           = dlsym(cuda_dlp, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr  = dlsym(cuda_dlp, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = dlsym(cuda_dlp, "cuDevicePrimaryCtxRelease");
    cuInitPtr                    = dlsym(cuda_dlp, "cuInit");
    cuGetErrorStringPtr          = dlsym(cuda_dlp, "cuGetErrorString");
    cuCtxPopCurrentPtr           = dlsym(cuda_dlp, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr          = dlsym(cuda_dlp, "cuCtxPushCurrent");
    cuCtxSynchronizePtr          = dlsym(cuda_dlp, "cuCtxSynchronize");
    cuCtxAttachPtr               = dlsym(cuda_dlp, "cuCtxAttach");
    cuCtxDetachPtr               = dlsym(cuda_dlp, "cuCtxDetach");
    cuDeviceGetAttributePtr      = dlsym(cuda_dlp, "cuDeviceGetAttribute");

    int cuda_not_initialized = (
        !cuCtxSetCurrentPtr           ||
        !cuCtxGetCurrentPtr           ||
        !cuCtxDestroyPtr              ||
        !cuCtxCreatePtr               ||
        !cuCtxGetDevicePtr            ||
        !cuDeviceGetPtr               ||
        !cuDeviceGetCountPtr          ||
        !cuDeviceGetNamePtr           ||
        !cuDevicePrimaryCtxRetainPtr  ||
        !cuDevicePrimaryCtxReleasePtr ||
        !cuInitPtr                    ||
        !cuGetErrorStringPtr          ||
        !cuCtxPopCurrentPtr           ||
        !cuCtxPushCurrentPtr          ||
        !cuCtxSynchronizePtr          ||
        !cuCtxAttachPtr               ||
        !cuCtxDetachPtr               ||
        !cuDeviceGetAttributePtr
    );

    papi_errno = (cuda_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading cuda driver symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

static int
unload_cuda_driver_sym(void)
{
    if (cuda_dlp == NULL) {
        return PAPI_OK;
    }

    cuCtxSetCurrentPtr           = NULL;
    cuCtxGetCurrentPtr           = NULL;
    cuCtxDestroyPtr              = NULL;
    cuCtxCreatePtr               = NULL;
    cuCtxGetDevicePtr            = NULL;
    cuDeviceGetPtr               = NULL;
    cuDeviceGetCountPtr          = NULL;
    cuDeviceGetNamePtr           = NULL;
    cuDevicePrimaryCtxRetainPtr  = NULL;
    cuDevicePrimaryCtxReleasePtr = NULL;
    cuInitPtr                    = NULL;
    cuGetErrorStringPtr          = NULL;
    cuCtxPopCurrentPtr           = NULL;
    cuCtxPushCurrentPtr          = NULL;
    cuCtxSynchronizePtr          = NULL;
    cuCtxAttachPtr               = NULL;
    cuCtxDetachPtr               = NULL;
    cuDeviceGetAttributePtr      = NULL;

    dlclose(cuda_dlp);

    return PAPI_OK;
}

static int
load_cuda_runtime_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = "libcudart.so";
    char *cuda_root = getenv("PAPI_CUDA_ROOT");
    if (cuda_root) {
        sprintf(pathname, "%s/lib64/libcudart.so", cuda_root);
    }

    cu_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (cu_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    cudaGetDevicePtr           = dlsym(cu_dlp, "cudaGetDevice");
    cudaGetDeviceCountPtr      = dlsym(cu_dlp, "cudaGetDeviceCount");
    cudaGetDevicePropertiesPtr = dlsym(cu_dlp, "cudaGetDeviceProperties");
    cudaDeviceGetAttributePtr  = dlsym(cu_dlp, "cudaDeviceGetAttribute");
    cudaSetDevicePtr           = dlsym(cu_dlp, "cudaSetDevice");
    cudaFreePtr                = dlsym(cu_dlp, "cudaFree");
    cudaDriverGetVersionPtr    = dlsym(cu_dlp, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr   = dlsym(cu_dlp, "cudaRuntimeGetVersion");

    int cu_not_initialized = (
        !cudaGetDevicePtr           ||
        !cudaGetDeviceCountPtr      ||
        !cudaGetDevicePropertiesPtr ||
        !cudaDeviceGetAttributePtr  ||
        !cudaSetDevicePtr           ||
        !cudaFreePtr                ||
        !cudaDriverGetVersionPtr    ||
        !cudaRuntimeGetVersionPtr
    );

    papi_errno = (cu_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading cuda runtime symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

static int
unload_cuda_runtime_sym(void)
{
    if (cu_dlp == NULL) {
        return PAPI_OK;
    }

    cudaGetDevicePtr           = NULL;
    cudaGetDeviceCountPtr      = NULL;
    cudaGetDevicePropertiesPtr = NULL;
    cudaDeviceGetAttributePtr  = NULL;
    cudaSetDevicePtr           = NULL;
    cudaFreePtr                = NULL;
    cudaDriverGetVersionPtr    = NULL;
    cudaRuntimeGetVersionPtr   = NULL;

    dlclose(cu_dlp);

    return PAPI_OK;
}

static int
load_cupti_common_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = "libcupti.so";
    char *cuda_root = getenv("PAPI_CUDA_ROOT");
    if (cuda_root) {
        sprintf(pathname, "%s/extras/CUPTI/lib64/libcupti.so", cuda_root);
    }

    cupti_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (cupti_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    cuptiGetVersionPtr = dlsym(cupti_dlp, "cuptiGetVersion");

    int cupti_not_initialized = !cuptiGetVersionPtr;
    papi_errno = (cupti_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading cupti common symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

static int
unload_cupti_common_sym(void)
{
    if (cupti_dlp == NULL) {
        return PAPI_OK;
    }

    cuptiGetVersionPtr = NULL;

    dlclose(cupti_dlp);

    return PAPI_OK;
}

int
cuptic_init(void)
{
    int papi_errno;

    papi_errno = load_cuda_driver_sym();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = load_cuda_runtime_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = load_cupti_common_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    const char *errstr;
    CUresult cu_errno = cuGetErrorStringPtr(CUDA_SUCCESS, &errstr);
    if (cu_errno == CUDA_ERROR_NOT_INITIALIZED) {
        cu_errno = cuInitPtr(0);
        if (cu_errno != CUDA_SUCCESS) {
            sprintf(error_string, "Error while initializing cupti layer.");
            papi_errno = PAPI_EMISC;
            goto fn_exit;
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_cupti_common_sym();
    unload_cuda_runtime_sym();
    unload_cuda_driver_sym();
    goto fn_exit;
}

int
cuptic_shutdown(void)
{
    int papi_errno;

    papi_errno = unload_cupti_common_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = unload_cuda_runtime_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = unload_cuda_driver_sym();

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

static int
get_device_count(int *count)
{
    int papi_errno = PAPI_OK;

    cudaError_t cuda_errno = cudaGetDeviceCountPtr(count);
    if (cuda_errno != cudaSuccess) {
        papi_errno = PAPI_ECMP;
    }

    return papi_errno;
}

int
cuptic_info_create(cuptic_info_t *info)
{
    int papi_errno = PAPI_OK;
    int count;

    if (*info != NULL) {
        return PAPI_EINVAL;
    }

    papi_errno = get_device_count(&count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    *info = papi_calloc(count, sizeof(**info));
    if ((*info) == NULL) {
        papi_errno = PAPI_ENOMEM;
    }

    return papi_errno;
}

int
cuptic_info_update(cuptic_info_t info, const char **events, int num_events)
{
    int papi_errno = PAPI_OK;

    if (info == NULL) {
        return PAPI_OK;
    }

    CUcontext user_ctx;
    CUresult cu_errno = cuCtxGetCurrentPtr(&user_ctx);
    if (cu_errno != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }

    int user_device_id;
    cudaError_t cuda_errno = cudaGetDevicePtr(&user_device_id);
    if (cuda_errno != cudaSuccess) {
        return PAPI_EMISC;
    }

    int i, device_id;
    for (i = 0; i < num_events; ++i) {
        papi_errno = cuptic_evt_get_device(events[i], &device_id);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        CUdevice device;
        cu_errno = cuDeviceGetPtr(&device, device_id);
        if (cu_errno != CUDA_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        if (user_ctx && info[device_id].ctx == NULL && device_id == user_device_id) {
            info[device_id].ctx = user_ctx;
            cu_errno = cuCtxAttachPtr(&info[device_id].ctx, 0);
            if (cu_errno != CUDA_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }
        } else if (info[device_id].ctx == NULL && info[device_id].primary == NULL) {
            cuda_errno = cudaSetDevicePtr(device_id);
            if (cuda_errno != cudaSuccess) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            cuda_errno = cudaFreePtr(NULL);
            if (cuda_errno != cudaSuccess) {
                papi_errno = PAPI_EMISC;
                goto fn_fail_restore_device;
            }

            cu_errno = cuCtxGetCurrentPtr(&info[device_id].primary);
            if (cu_errno != CUDA_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail_restore_device;
            }

            cuda_errno = cudaSetDevicePtr(user_device_id);
            if (cuda_errno != cudaSuccess) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            info[device_id].ctx = info[device_id].primary;
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
  fn_fail_restore_device:
    cudaSetDevicePtr(user_device_id);
    goto fn_fail;
}

int
cuptic_info_destroy(cuptic_info_t info)
{
    int papi_errno;

    if (info == NULL) {
        return PAPI_EINVAL;
    }

    int count;
    papi_errno = get_device_count(&count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int i;
    for (i = 0; i < count; ++i) {
        CUdevice device;
        cuDeviceGetPtr(&device, i);
        if (info[i].ctx != info[i].primary) {
            cuCtxDetachPtr(info[i].ctx);
        }
    }

    papi_free(info);

    return papi_errno;
}

#if defined(API_PERFWORKS) && defined(API_EVENTS)
int
cuptic_api_fallback_event(void)
{
    int device_count;
    int cc_maj, cc_min;
    static int retval_set;
    static int retval;
    CUresult cu_errno;

    if (retval_set) {
        return retval;
    }

    cudaError_t cuda_errno = cudaGetDeviceCountPtr(&device_count);
    if (cuda_errno != cudaSuccess) {
        return 0;
    }

    int i;
    for (i = 0; i < device_count; ++i) {
        CUdevice device;
        cu_errno = cuDeviceGetPtr(&device, i);
        if (cu_errno != CUDA_SUCCESS) {
            return 0;
        }

        cu_errno = cuDeviceGetAttributePtr(&cc_maj, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        if (cu_errno != CUDA_SUCCESS) {
            return 0;
        }

        cu_errno = cuDeviceGetAttributePtr(&cc_min, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        if (cu_errno != CUDA_SUCCESS) {
            return 0;
        }

#if CUPTI_VERSION == 13
        if (cc_maj == 7 && cc_min == 0) {
            /* fallback if compute capability of any GPU is 7.0 */
            retval_set = retval = 1;
            return retval;
        }
#endif
    }

    retval_set = 1;
    return retval;
}
#else
int
cuptic_api_fallback_event(void)
{
    return 0;
}
#endif

int
cuptic_api_check_version(void)
{
    cudaError_t cuda_errno;
    CUptiResult cupti_errno;
    int runtimeversion, driverversion;
    unsigned int cuptiversion;

    cuda_errno = cudaRuntimeGetVersionPtr(&runtimeversion);
    if (cuda_errno != cudaSuccess) {
        return PAPI_EMISC;
    }

    cuda_errno = cudaDriverGetVersionPtr(&driverversion);
    if (cuda_errno != cudaSuccess) {
        return PAPI_EMISC;
    }

    cupti_errno = cuptiGetVersionPtr(&cuptiversion);
    if (cupti_errno != CUPTI_SUCCESS) {
        return PAPI_EMISC;
    }

    if (runtimeversion != CUDA_VERSION || cuptiversion != CUPTI_API_VERSION ||
        (driverversion < runtimeversion)) {
        return PAPI_ECMP;
    }

    return PAPI_OK;
}

int
cuptic_dev_get_map(const char **events, int num_events, cuptic_bitmap_t *bitmap)
{
    cuptic_bitmap_t device_map_acq = 0;

    int i;
    for (i = 0; i < num_events; ++i) {
        int device_id;
        cuptic_evt_get_device(events[i], &device_id);
        device_map_acq |= (1 << device_id);
    }
    *bitmap = device_map_acq;

    return PAPI_OK;
}

int
cuptic_dev_acquire(cuptic_bitmap_t bitmap)
{
    cuptic_bitmap_t device_map_acq = bitmap;

    if (device_map_acq & global_device_map) {
        return PAPI_ECNFLCT;
    }
    global_device_map |= device_map_acq;

    return PAPI_OK;
}

int
cuptic_dev_release(cuptic_bitmap_t bitmap)
{
    cuptic_bitmap_t device_map_rel = bitmap;

    if ((device_map_rel & global_device_map) != device_map_rel) {
        return PAPI_EMISC;
    }
    global_device_map &= ~device_map_rel;

    return PAPI_OK;
}

int
cuptic_dev_get_count(cuptic_bitmap_t device_map, int *num_devices)
{
    *num_devices = 0;
    int i;
    for (i = 0; i < (int) sizeof(device_map) * 8; ++i) {
        if ((device_map >> i) & 0x1) {
            ++(*num_devices);
        }
    }
    return PAPI_OK;
}

int
cuptic_dev_get_id(cuptic_bitmap_t device_map, int dev_count, int *device_id)
{
    int i, count = 0;

    for (i = 0; i < (int) sizeof(device_map) * 8; ++i) {
        if (((device_map >> i) & 0x1) && (count++ == dev_count)) {
            break;
        }
    }

    *device_id = i;
    return PAPI_OK;
}

int
cuptic_evt_get_device(const char *string, int *device)
{
    int i = 0;

    char *tmp = papi_strdup(string);
    char *ptr = strstr(tmp, "device=");
    if (ptr == NULL) {
        return PAPI_EINVAL;
    }

    ptr += strlen("device=");
    while (*(ptr + i) != ':' && *(ptr + i) != '\0') {
        ++i;
    }

    *(ptr + i) = '\0';
    *device = (int) strtol(ptr, NULL, 10); 

    papi_free(tmp);
    return PAPI_OK;
}

int
cuptic_err_get_last(const char **err_string)
{
    *err_string = error_string;
    return PAPI_OK;
}
