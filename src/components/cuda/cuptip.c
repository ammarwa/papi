#if defined(API_PERFWORKS)
#include <cupti_target.h>
#include <cupti_profiler_target.h>
#include <nvperf_host.h>
#include <nvperf_target.h>
#include <nvperf_cuda_host.h>
#include <dlfcn.h>
#include "cuptip.h"
#include "htable.h"

NVPA_Status (*NVPW_GetSupportedChipNamesPtr)(NVPW_GetSupportedChipNames_Params *params);
NVPA_Status (*NVPW_MetricsContext_DestroyPtr)(NVPW_MetricsContext_Destroy_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetMetricNames_BeginPtr)(NVPW_MetricsContext_GetMetricNames_Begin_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetMetricNames_EndPtr)(NVPW_MetricsContext_GetMetricNames_End_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetMetricProperties_BeginPtr)(NVPW_MetricsContext_GetMetricProperties_Begin_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetMetricProperties_EndPtr)(NVPW_MetricsContext_GetMetricProperties_End_Params *params);
NVPA_Status (*NVPW_InitializeHostPtr)(NVPW_InitializeHost_Params *params);
NVPA_Status (*NVPW_CUDA_MetricsContext_CreatePtr)(NVPW_CUDA_MetricsContext_Create_Params *params);
NVPA_Status (*NVPW_CUDA_RawMetricsConfig_CreatePtr)(NVPW_CUDA_RawMetricsConfig_Create_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_DestroyPtr)(NVPW_RawMetricsConfig_Destroy_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_BeginPassGroupPtr)(NVPW_RawMetricsConfig_BeginPassGroup_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_EndPassGroupPtr)(NVPW_RawMetricsConfig_EndPassGroup_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_AddMetricsPtr)(NVPW_RawMetricsConfig_AddMetrics_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_GenerateConfigImagePtr)(NVPW_RawMetricsConfig_GenerateConfigImage_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_GetConfigImagePtr)(NVPW_RawMetricsConfig_GetConfigImage_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_GetNumPassesPtr)(NVPW_RawMetricsConfig_GetNumPasses_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_SetCounterAvailabilityPtr)(NVPW_RawMetricsConfig_SetCounterAvailability_Params *params);
NVPA_Status (*NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr)(NVPW_RawMetricsConfig_IsAddMetricsPossible_Params *params);
NVPA_Status (*NVPW_CounterDataBuilder_CreatePtr)(NVPW_CounterDataBuilder_Create_Params *params);
NVPA_Status (*NVPW_CounterDataBuilder_DestroyPtr)(NVPW_CounterDataBuilder_Destroy_Params *params);
NVPA_Status (*NVPW_CounterDataBuilder_AddMetricsPtr)(NVPW_CounterDataBuilder_AddMetrics_Params *params);
NVPA_Status (*NVPW_CounterDataBuilder_GetCounterDataPrefixPtr)(NVPW_CounterDataBuilder_GetCounterDataPrefix_Params *params);
NVPA_Status (*NVPW_CounterData_GetNumRangesPtr)(NVPW_CounterData_GetNumRanges_Params *params);
NVPA_Status (*NVPW_Profiler_CounterData_GetRangeDescriptionsPtr)(NVPW_Profiler_CounterData_GetRangeDescriptions_Params *params);
NVPA_Status (*NVPW_MetricsContext_SetCounterDataPtr)(NVPW_MetricsContext_SetCounterData_Params *params);
NVPA_Status (*NVPW_MetricsContext_EvaluateToGpuValuesPtr)(NVPW_MetricsContext_EvaluateToGpuValues_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetCounterNames_BeginPtr)(NVPW_MetricsContext_GetCounterNames_Begin_Params *params);
NVPA_Status (*NVPW_MetricsContext_GetCounterNames_EndPtr)(NVPW_MetricsContext_GetCounterNames_End_Params *params);

CUptiResult (*cuptiDeviceGetChipNamePtr)(CUpti_Device_GetChipName_Params *params);
CUptiResult (*cuptiProfilerInitializePtr)(CUpti_Profiler_Initialize_Params *params);
CUptiResult (*cuptiProfilerDeInitializePtr)(CUpti_Profiler_DeInitialize_Params *params);
CUptiResult (*cuptiProfilerCounterDataImageCalculateSizePtr)(CUpti_Profiler_CounterDataImage_CalculateSize_Params *params);
CUptiResult (*cuptiProfilerCounterDataImageInitializePtr)(CUpti_Profiler_CounterDataImage_Initialize_Params *params);
CUptiResult (*cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr)(CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params *params);
CUptiResult (*cuptiProfilerCounterDataImageInitializeScratchBufferPtr)(CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params *params);
CUptiResult (*cuptiProfilerBeginSessionPtr)(CUpti_Profiler_BeginSession_Params *params);
CUptiResult (*cuptiProfilerSetConfigPtr)(CUpti_Profiler_SetConfig_Params *params);
CUptiResult (*cuptiProfilerBeginPassPtr)(CUpti_Profiler_BeginPass_Params *params);
CUptiResult (*cuptiProfilerEnableProfilingPtr)(CUpti_Profiler_EnableProfiling_Params *params);
CUptiResult (*cuptiProfilerPushRangePtr)(CUpti_Profiler_PushRange_Params *params);
CUptiResult (*cuptiProfilerPopRangePtr)(CUpti_Profiler_PopRange_Params *params);
CUptiResult (*cuptiProfilerDisableProfilingPtr)(CUpti_Profiler_DisableProfiling_Params *params);
CUptiResult (*cuptiProfilerEndPassPtr)(CUpti_Profiler_EndPass_Params *params);
CUptiResult (*cuptiProfilerFlushCounterDataPtr)(CUpti_Profiler_FlushCounterData_Params *params);
CUptiResult (*cuptiProfilerUnsetConfigPtr)(CUpti_Profiler_UnsetConfig_Params *params);
CUptiResult (*cuptiProfilerEndSessionPtr)(CUpti_Profiler_EndSession_Params *params);
CUptiResult (*cuptiProfilerGetCounterAvailabilityPtr)(CUpti_Profiler_GetCounterAvailability_Params *params);
CUptiResult (*cuptiFinalizePtr)(void);

struct cuptip_ctl {
};

static int load_cupti_sym(void);
static int load_nvpw_sym(void);
static int unload_cupti_sym(void);
static int unload_nvpw_sym(void);
static int initialize_cupti_profiler_api(void);
static int initialize_perfworks_api(void);

int
cuptip_init(void)
{
    int papi_errno = PAPI_OK;

    papi_errno = load_cupti_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = load_nvpw_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptic_api_check_version();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "CUDA runtime driver or CUPTI library version mismatch.");
        goto fn_fail;
    }

    int count;
    cudaError_t cuda_errno = cudaGetDeviceCountPtr(&count);
    if (cuda_errno != cudaSuccess) {
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    if (count < 1) {
        sprintf(error_string, "No devices found in the system.");
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    papi_errno = initialize_cupti_profiler_api();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Unable to initialize PERFWORKS API.");
        goto fn_fail;
    }

    papi_errno = initialize_perfworks_api();
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Unable to initialize CUPTI API.");
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    unload_nvpw_sym();
    unload_cupti_sym();
    goto fn_exit;
}

int
load_cupti_sym(void)
{
    int papi_errno = PAPI_OK;

    /*
     * cupti_dlp is initialized in cuptic.c and exported
     */
    cuptiDeviceGetChipNamePtr                                  = dlsym(cupti_dlp, "cuptiDeviceGetChipName");
    cuptiProfilerInitializePtr                                 = dlsym(cupti_dlp, "cuptiProfilerInitialize");
    cuptiProfilerDeInitializePtr                               = dlsym(cupti_dlp, "cuptiProfilerDeInitialize");
    cuptiProfilerCounterDataImageCalculateSizePtr              = dlsym(cupti_dlp, "cuptiProfilerCounterDataImageCalculateSize");
    cuptiProfilerCounterDataImageInitializePtr                 = dlsym(cupti_dlp, "cuptiProfilerCounterDataImageInitialize");
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = dlsym(cupti_dlp, "cuptiProfilerCounterDataImageCalculateScratchBufferSize");
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr    = dlsym(cupti_dlp, "cuptiProfilerCounterDataImageInitializeScratchBuffer");
    cuptiProfilerBeginSessionPtr                               = dlsym(cupti_dlp, "cuptiProfilerBeginSession");
    cuptiProfilerSetConfigPtr                                  = dlsym(cupti_dlp, "cuptiProfilerSetConfig");
    cuptiProfilerBeginPassPtr                                  = dlsym(cupti_dlp, "cuptiProfilerBeginPass");
    cuptiProfilerEnableProfilingPtr                            = dlsym(cupti_dlp, "cuptiProfilerEnableProfiling");
    cuptiProfilerPushRangePtr                                  = dlsym(cupti_dlp, "cuptiProfilerPushRange");
    cuptiProfilerPopRangePtr                                   = dlsym(cupti_dlp, "cuptiProfilerPopRange");
    cuptiProfilerDisableProfilingPtr                           = dlsym(cupti_dlp, "cuptiProfilerDisableProfiling");
    cuptiProfilerEndPassPtr                                    = dlsym(cupti_dlp, "cuptiProfilerEndPass");
    cuptiProfilerFlushCounterDataPtr                           = dlsym(cupti_dlp, "cuptiProfilerFlushCounterData");
    cuptiProfilerUnsetConfigPtr                                = dlsym(cupti_dlp, "cuptiProfilerUnsetConfig");
    cuptiProfilerEndSessionPtr                                 = dlsym(cupti_dlp, "cuptiProfilerEndSession");
    cuptiProfilerGetCounterAvailabilityPtr                     = dlsym(cupti_dlp, "cuptiProfilerGetCounterAvailability");
    cuptiFinalizePtr                                           = dlsym(cupti_dlp, "cuptiFinalize");

    int cupti_not_initialized = (
        !cuptiDeviceGetChipNamePtr                                  ||
        !cuptiProfilerInitializePtr                                 ||
        !cuptiProfilerDeInitializePtr                               ||
        !cuptiProfilerCounterDataImageCalculateSizePtr              ||
        !cuptiProfilerCounterDataImageInitializePtr                 ||
        !cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr ||
        !cuptiProfilerCounterDataImageInitializeScratchBufferPtr    ||
        !cuptiProfilerBeginSessionPtr                               ||
        !cuptiProfilerSetConfigPtr                                  ||
        !cuptiProfilerBeginPassPtr                                  ||
        !cuptiProfilerEnableProfilingPtr                            ||
        !cuptiProfilerPushRangePtr                                  ||
        !cuptiProfilerPopRangePtr                                   ||
        !cuptiProfilerDisableProfilingPtr                           ||
        !cuptiProfilerEndPassPtr                                    ||
        !cuptiProfilerFlushCounterDataPtr                           ||
        !cuptiProfilerUnsetConfigPtr                                ||
        !cuptiProfilerEndSessionPtr                                 ||
        !cuptiProfilerGetCounterAvailabilityPtr                     ||
        !cuptiFinalizePtr
    );

    papi_errno = (cupti_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading cupti symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

static void *nvpw_dlp;

int
load_nvpw_sym(void)
{
    int papi_errno = PAPI_OK;

    char pathname[PATH_MAX] = "libnvperf_host.so";
    char *cuda_root = getenv("PAPI_CUDA_ROOT");
    if (cuda_root) {
        sprintf(pathname, "%s/extras/CUPTI/lib64/libnvperf_host.so", cuda_root);
    }

    nvpw_dlp = dlopen(pathname, RTLD_NOW | RTLD_GLOBAL);
    if (nvpw_dlp == NULL) {
        sprintf(error_string, "%s", dlerror());
        goto fn_fail;
    }

    NVPW_InitializeHostPtr                            = dlsym(nvpw_dlp, "NVPW_InitializeHost");
    NVPW_GetSupportedChipNamesPtr                     = dlsym(nvpw_dlp, "NVPW_GetSupportedChipNames");
    NVPW_MetricsContext_DestroyPtr                    = dlsym(nvpw_dlp, "NVPW_MetricsContext_Destroy");
    NVPW_MetricsContext_GetMetricNames_BeginPtr       = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetMetricNames_Begin");
    NVPW_MetricsContext_GetMetricNames_EndPtr         = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetMetricNames_End");
    NVPW_MetricsContext_GetMetricProperties_BeginPtr  = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetMetricProperties_Begin");
    NVPW_MetricsContext_GetMetricProperties_EndPtr    = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetMetricProperties_End");
    NVPW_MetricsContext_SetCounterDataPtr             = dlsym(nvpw_dlp, "NVPW_MetricsContext_SetCounterData");
    NVPW_MetricsContext_EvaluateToGpuValuesPtr        = dlsym(nvpw_dlp, "NVPW_MetricsContext_EvaluateToGpuValues");
    NVPW_MetricsContext_GetCounterNames_BeginPtr      = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetCounterNames_Begin");
    NVPW_MetricsContext_GetCounterNames_EndPtr        = dlsym(nvpw_dlp, "NVPW_MetricsContext_GetCounterNames_End");
    NVPW_CUDA_MetricsContext_CreatePtr                = dlsym(nvpw_dlp, "NVPW_CUDA_MetricsContext_Create");
    NVPW_CUDA_RawMetricsConfig_CreatePtr              = dlsym(nvpw_dlp, "NVPW_CUDA_RawMetricsConfig_Create");
    NVPW_RawMetricsConfig_DestroyPtr                  = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_Destroy");
    NVPW_RawMetricsConfig_BeginPassGroupPtr           = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_BeginPassGroup");
    NVPW_RawMetricsConfig_EndPassGroupPtr             = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_EndPassGroup");
    NVPW_RawMetricsConfig_AddMetricsPtr               = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_AddMetrics");
    NVPW_RawMetricsConfig_GenerateConfigImagePtr      = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_GenerateConfigImage");
    NVPW_RawMetricsConfig_GetConfigImagePtr           = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_GetConfigImage");
    NVPW_RawMetricsConfig_GetNumPassesPtr             = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_GetNumPasses");
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr   = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_SetCounterAvailability");
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr     = dlsym(nvpw_dlp, "NVPW_RawMetricsConfig_IsAddMetricsPossible");
    NVPW_CounterDataBuilder_CreatePtr                 = dlsym(nvpw_dlp, "NVPW_CounterDataBuilder_Create");
    NVPW_CounterDataBuilder_DestroyPtr                = dlsym(nvpw_dlp, "NVPW_CounterDataBuilder_Destroy");
    NVPW_CounterDataBuilder_AddMetricsPtr             = dlsym(nvpw_dlp, "NVPW_CounterDataBuilder_AddMetrics");
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr   = dlsym(nvpw_dlp, "NVPW_CounterDataBuilder_GetCounterDataPrefix");
    NVPW_CounterData_GetNumRangesPtr                  = dlsym(nvpw_dlp, "NVPW_CounterData_GetNumRanges");
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr = dlsym(nvpw_dlp, "NVPW_Profiler_CounterData_GetRangeDescriptions");

    int nvpw_not_initialized = (
        !NVPW_InitializeHostPtr                              ||
        !NVPW_GetSupportedChipNamesPtr                       ||
        !NVPW_MetricsContext_DestroyPtr                      ||
        !NVPW_MetricsContext_GetMetricNames_BeginPtr         ||
        !NVPW_MetricsContext_GetMetricNames_EndPtr           ||
        !NVPW_MetricsContext_GetMetricProperties_BeginPtr    ||
        !NVPW_MetricsContext_GetMetricProperties_EndPtr      ||
        !NVPW_MetricsContext_SetCounterDataPtr               ||
        !NVPW_MetricsContext_EvaluateToGpuValuesPtr          ||
        !NVPW_MetricsContext_GetCounterNames_BeginPtr        ||
        !NVPW_MetricsContext_GetCounterNames_EndPtr          ||
        !NVPW_CUDA_MetricsContext_CreatePtr                  ||
        !NVPW_CUDA_RawMetricsConfig_CreatePtr                ||
        !NVPW_RawMetricsConfig_DestroyPtr                    ||
        !NVPW_RawMetricsConfig_BeginPassGroupPtr             ||
        !NVPW_RawMetricsConfig_EndPassGroupPtr               ||
        !NVPW_RawMetricsConfig_AddMetricsPtr                 ||
        !NVPW_RawMetricsConfig_GenerateConfigImagePtr        ||
        !NVPW_RawMetricsConfig_GetConfigImagePtr             ||
        !NVPW_RawMetricsConfig_GetNumPassesPtr               ||
        !NVPW_RawMetricsConfig_SetCounterAvailabilityPtr     ||
        !NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr       ||
        !NVPW_CounterDataBuilder_CreatePtr                   ||
        !NVPW_CounterDataBuilder_DestroyPtr                  ||
        !NVPW_CounterDataBuilder_AddMetricsPtr               ||
        !NVPW_CounterDataBuilder_GetCounterDataPrefixPtr     ||
        !NVPW_CounterData_GetNumRangesPtr                    ||
        !NVPW_Profiler_CounterData_GetRangeDescriptionsPtr
    );

    papi_errno = (nvpw_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading nvpw symbols.");
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    papi_errno = PAPI_ENOSUPP;
    goto fn_exit;
}

int
initialize_cupti_profiler_api(void)
{
    CUpti_Profiler_Initialize_Params profilerInitializeParams = { CUpti_Profiler_Initialize_Params_STRUCT_SIZE, NULL };
    if (cuptiProfilerInitializePtr(&profilerInitializeParams) != CUPTI_SUCCESS) {
        return PAPI_ESYS;
    }
    return PAPI_OK;
}

int
initialize_perfworks_api(void)
{
    NVPW_InitializeHost_Params perfInitHostParams = { NVPW_InitializeHost_Params_STRUCT_SIZE, NULL };
    if (NVPW_InitializeHostPtr(&perfInitHostParams) != NVPA_STATUS_SUCCESS) {
        return PAPI_ESYS;
    }
    return PAPI_OK;
}

int
unload_cupti_sym(void)
{
    if (cupti_dlp == NULL) {
        return PAPI_OK;
    }

    cuptiDeviceGetChipNamePtr                                  = NULL;
    cuptiProfilerInitializePtr                                 = NULL;
    cuptiProfilerDeInitializePtr                               = NULL;
    cuptiProfilerCounterDataImageCalculateSizePtr              = NULL;
    cuptiProfilerCounterDataImageInitializePtr                 = NULL;
    cuptiProfilerCounterDataImageCalculateScratchBufferSizePtr = NULL;
    cuptiProfilerCounterDataImageInitializeScratchBufferPtr    = NULL;
    cuptiProfilerBeginSessionPtr                               = NULL;
    cuptiProfilerSetConfigPtr                                  = NULL;
    cuptiProfilerBeginPassPtr                                  = NULL;
    cuptiProfilerEnableProfilingPtr                            = NULL;
    cuptiProfilerPushRangePtr                                  = NULL;
    cuptiProfilerPopRangePtr                                   = NULL;
    cuptiProfilerDisableProfilingPtr                           = NULL;
    cuptiProfilerEndPassPtr                                    = NULL;
    cuptiProfilerFlushCounterDataPtr                           = NULL;
    cuptiProfilerUnsetConfigPtr                                = NULL;
    cuptiProfilerEndSessionPtr                                 = NULL;
    cuptiProfilerGetCounterAvailabilityPtr                     = NULL;
    cuptiFinalizePtr                                           = NULL;

    /*
     * cupti_dlp is finalized in cuptic.c
     */

    return PAPI_OK;
}

int
unload_nvpw_sym(void)
{
    if (nvpw_dlp == NULL) {
        return PAPI_OK;
    }

    NVPW_InitializeHostPtr                              = NULL;
    NVPW_GetSupportedChipNamesPtr                       = NULL;
    NVPW_MetricsContext_DestroyPtr                      = NULL;
    NVPW_MetricsContext_GetMetricNames_BeginPtr         = NULL;
    NVPW_MetricsContext_GetMetricNames_EndPtr           = NULL;
    NVPW_MetricsContext_GetMetricProperties_BeginPtr    = NULL;
    NVPW_MetricsContext_GetMetricProperties_EndPtr      = NULL;
    NVPW_MetricsContext_SetCounterDataPtr               = NULL;
    NVPW_MetricsContext_EvaluateToGpuValuesPtr          = NULL;
    NVPW_MetricsContext_GetCounterNames_BeginPtr        = NULL;
    NVPW_MetricsContext_GetCounterNames_EndPtr          = NULL;
    NVPW_CUDA_MetricsContext_CreatePtr                  = NULL;
    NVPW_CUDA_RawMetricsConfig_CreatePtr                = NULL;
    NVPW_RawMetricsConfig_DestroyPtr                    = NULL;
    NVPW_RawMetricsConfig_BeginPassGroupPtr             = NULL;
    NVPW_RawMetricsConfig_EndPassGroupPtr               = NULL;
    NVPW_RawMetricsConfig_AddMetricsPtr                 = NULL;
    NVPW_RawMetricsConfig_GenerateConfigImagePtr        = NULL;
    NVPW_RawMetricsConfig_GetConfigImagePtr             = NULL;
    NVPW_RawMetricsConfig_GetNumPassesPtr               = NULL;
    NVPW_RawMetricsConfig_SetCounterAvailabilityPtr     = NULL;
    NVPW_RawMetricsConfig_IsAddMetricsPossiblePtr       = NULL;
    NVPW_CounterDataBuilder_CreatePtr                   = NULL;
    NVPW_CounterDataBuilder_DestroyPtr                  = NULL;
    NVPW_CounterDataBuilder_AddMetricsPtr               = NULL;
    NVPW_CounterDataBuilder_GetCounterDataPrefixPtr     = NULL;
    NVPW_CounterData_GetNumRangesPtr                    = NULL;
    NVPW_Profiler_CounterData_GetRangeDescriptionsPtr   = NULL;

    dlclose(nvpw_dlp);

    return PAPI_OK;
}

int
cuptip_ctl_create(const char **events, int num_events, cuptic_info_t info, cuptip_ctl_t *ctl)
{
    int papi_errno = PAPI_OK;
    struct cuptip_ctl *cupti_ctl;

    _papi_hwi_lock(_cuda_lock);

  fn_exit:
    _papi_hwi_unlock(_cuda_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_ctl_destroy(cuptip_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

    _papi_hwi_lock(_cuda_lock);

  fn_exit:
    _papi_hwi_unlock(_cuda_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_ctl_start(cuptip_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_ctl_stop(cuptip_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_ctl_read(cuptip_ctl_t ctl, long long **counters)
{
    int papi_errno = PAPI_OK;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_ctl_reset(cuptip_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptip_shutdown(void)
{
    return PAPI_OK;
}

int
cuptip_evt_enum(unsigned int *event_code, int modifier)
{
    return PAPI_OK;
}

int
cuptip_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    return PAPI_OK;
}

int
cuptip_evt_name_to_code(const char *name, unsigned int *event_code)
{
    return PAPI_OK;
}

int
cuptip_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    return PAPI_OK;
}
#endif
