#include <dlfcn.h>
#include "cuptie.h"
#include "htable.h"

#define CUPTIE_CTL_CREATED 0x1
#define CUPTIE_CTL_STARTED 0x2

CUptiResult (*cuptiDeviceEnumMetricsPtr)(CUdevice, size_t *, CUpti_MetricID *);
CUptiResult (*cuptiDeviceGetEventDomainAttributePtr)(CUdevice, CUpti_EventDomainID, CUpti_EventDomainAttribute, size_t *, void *);
CUptiResult (*cuptiDeviceGetNumMetricsPtr)(CUdevice, uint32_t *);
CUptiResult (*cuptiDeviceEnumEventDomainsPtr)(CUdevice, size_t *, CUpti_EventDomainID *);
CUptiResult (*cuptiDeviceGetNumEventDomainsPtr)(CUdevice, uint32_t *);
CUptiResult (*cuptiEventGroupGetAttributePtr)(CUpti_EventGroup, CUpti_EventGroupAttribute, size_t *, void *);
CUptiResult (*cuptiEventGroupReadEventPtr)(CUpti_EventGroup, CUpti_ReadEventFlags, CUpti_EventID, size_t *, uint64_t *);
CUptiResult (*cuptiEventGroupSetAttributePtr)(CUpti_EventGroup, CUpti_EventGroupAttribute, size_t, void *);
CUptiResult (*cuptiEventGroupSetDisablePtr)(CUpti_EventGroupSet *);
CUptiResult (*cuptiEventGroupSetEnablePtr)(CUpti_EventGroupSet *);
CUptiResult (*cuptiEventGroupSetsCreatePtr)(CUcontext, size_t, CUpti_EventID *, CUpti_EventGroupSets **);
CUptiResult (*cuptiEventGroupSetsDestroyPtr)(CUpti_EventGroupSets *);
CUptiResult (*cuptiEventGroupAddEventPtr)(CUpti_EventGroup, CUpti_EventID);
CUptiResult (*cuptiEventGroupCreatePtr)(CUcontext, CUpti_EventGroup *, uint32_t);
CUptiResult (*cuptiEventGroupDestroyPtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupDisablePtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupEnablePtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGroupReadAllEventsPtr)(CUpti_EventGroup, CUpti_ReadEventFlags, size_t *, uint64_t *, size_t *, CUpti_EventID *,
                                               size_t *);
CUptiResult (*cuptiEventGroupResetAllEventsPtr)(CUpti_EventGroup);
CUptiResult (*cuptiEventGetIdFromNamePtr)(CUdevice device, const char *event_name, CUpti_EventID *event);
CUptiResult (*cuptiMetricCreateEventGroupSetsPtr)(CUcontext, size_t, CUpti_MetricID *, CUpti_EventGroupSets **);
CUptiResult (*cuptiMetricGetRequiredEventGroupSetsPtr)(CUcontext, CUpti_MetricID, CUpti_EventGroupSets **);
CUptiResult (*cuptiMetricGetAttributePtr)(CUpti_MetricID, CUpti_MetricAttribute, size_t *, void *);
CUptiResult (*cuptiMetricGetNumEventsPtr)(CUpti_MetricID, uint32_t *);
CUptiResult (*cuptiMetricGetValuePtr)(CUdevice, CUpti_MetricID, size_t, CUpti_EventID *, size_t, uint64_t *, uint64_t,
                                      CUpti_MetricValue *);
CUptiResult (*cuptiMetricEnumEventsPtr)(CUpti_MetricID, size_t *, CUpti_EventID *);
CUptiResult (*cuptiMetricGetIdFromNamePtr)(CUdevice device, const char *metric_name, CUpti_MetricID *metric);
CUptiResult (*cuptiGetTimestampPtr)(uint64_t *);
CUptiResult (*cuptiSetEventCollectionModePtr)(CUcontext, CUpti_EventCollectionMode);
CUptiResult (*cuptiEventDomainEnumEventsPtr)(CUpti_EventDomainID, size_t *, CUpti_EventID *);
CUptiResult (*cuptiEventDomainGetAttributePtr)(CUpti_EventDomainID, CUpti_EventDomainAttribute, size_t *, void *);
CUptiResult (*cuptiEventDomainGetNumEventsPtr)(CUpti_EventDomainID, uint32_t *);
CUptiResult (*cuptiEventGetAttributePtr)(CUpti_EventID, CUpti_EventAttribute, size_t *, void *);
CUptiResult (*cuptiGetResultStringPtr)(CUptiResult, const char **);
CUptiResult (*cuptiEnableKernelReplayModePtr)(CUcontext);
CUptiResult (*cuptiDisableKernelReplayModePtr)(CUcontext);

typedef struct {
    char name[PAPI_MIN_STR_LEN];
    int32_t cc_major;
    int32_t cc_minor;
    CUdevice cu_device;
    int device_id;
} device_t;

typedef struct {
    device_t *devices;
    int32_t count;
} device_table_t;

typedef struct {
    CUpti_EventID cupti_event_id;
    int64_t cupti_event_value;
} cupti_event_t;

typedef struct {
    CUpti_MetricID cupti_metric_id;
    CUpti_EventID *cupti_events_id;
    uint64_t *cupti_events_value;
    int32_t num_cupti_events;
    CUpti_MetricValueKind unit;
} cupti_metric_t;

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_2MAX_STR_LEN];
    uint32_t papi_event_id;
    CUpti_ActivityKind kind;
    device_t *device;
    union {
        cupti_event_t e;
        cupti_metric_t m;
    } u;
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int32_t count;
} ntv_event_table_t;

struct cuptie_ctl {
    int state;
    CUpti_EventID *cupti_events_id;
    int32_t num_cupti_events;
    int32_t *num_cupti_events_per_dev;
    uint32_t *papi_events_id;
    int32_t num_papi_events;
    cuptic_bitmap_t device_map;
    cuptic_info_t info;
    int64_t *counters;
    CUpti_EventGroupSets **event_group_sets;
    uint64_t start_time_ns;
    uint64_t stop_time_ns;
};

static void *htable;
static ntv_event_table_t ntv_event_table;
static ntv_event_table_t *ntv_event_table_p;
static device_table_t device_table;
static device_table_t *device_table_p;

static int load_cupti_sym(void);
static int unload_cupti_sym(void);
static int init_device_table(void);
static int init_event_table(void);
static int finalize_device_table(void);
static int finalize_event_table(void);

int
cuptie_init(void)
{
    int papi_errno = PAPI_OK;

    papi_errno = load_cupti_sym();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    int htable_errno = htable_init(&htable);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = init_device_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = init_event_table();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    finalize_event_table();
    finalize_device_table();
    htable_shutdown(htable);
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
    cuptiDeviceEnumMetricsPtr               = dlsym(cupti_dlp, "cuptiDeviceEnumMetrics");
    cuptiDeviceGetEventDomainAttributePtr   = dlsym(cupti_dlp, "cuptiDeviceGetEventDomainAttribute");
    cuptiDeviceGetNumMetricsPtr             = dlsym(cupti_dlp, "cuptiDeviceGetNumMetrics");
    cuptiDeviceEnumEventDomainsPtr          = dlsym(cupti_dlp, "cuptiDeviceEnumEventDomains");
    cuptiDeviceGetNumEventDomainsPtr        = dlsym(cupti_dlp, "cuptiDeviceGetNumEventDomains");
    cuptiEventGroupGetAttributePtr          = dlsym(cupti_dlp, "cuptiEventGroupGetAttribute");
    cuptiEventGroupReadEventPtr             = dlsym(cupti_dlp, "cuptiEventGroupReadEvent");
    cuptiEventGroupSetAttributePtr          = dlsym(cupti_dlp, "cuptiEventGroupSetAttribute");
    cuptiEventGroupSetDisablePtr            = dlsym(cupti_dlp, "cuptiEventGroupSetDisable");
    cuptiEventGroupSetEnablePtr             = dlsym(cupti_dlp, "cuptiEventGroupSetEnable");
    cuptiEventGroupSetsCreatePtr            = dlsym(cupti_dlp, "cuptiEventGroupSetsCreate");
    cuptiEventGroupSetsDestroyPtr           = dlsym(cupti_dlp, "cuptiEventGroupSetsDestroy");
    cuptiEventGroupAddEventPtr              = dlsym(cupti_dlp, "cuptiEventGroupAddEvent");
    cuptiEventGroupCreatePtr                = dlsym(cupti_dlp, "cuptiEventGroupCreate");
    cuptiEventGroupDestroyPtr               = dlsym(cupti_dlp, "cuptiEventGroupDestroy");
    cuptiEventGroupDisablePtr               = dlsym(cupti_dlp, "cuptiEventGroupDisable");
    cuptiEventGroupEnablePtr                = dlsym(cupti_dlp, "cuptiEventGroupEnable");
    cuptiEventGroupReadAllEventsPtr         = dlsym(cupti_dlp, "cuptiEventGroupResetAllEvents");
    cuptiEventGroupResetAllEventsPtr        = dlsym(cupti_dlp, "cuptiEventGroupResetAllEvents");
    cuptiEventGetIdFromNamePtr              = dlsym(cupti_dlp, "cuptiEventGetIdFromName");
    cuptiMetricCreateEventGroupSetsPtr      = dlsym(cupti_dlp, "cuptiMetricCreateEventGroupSets");
    cuptiMetricGetRequiredEventGroupSetsPtr = dlsym(cupti_dlp, "cuptiMetricGetRequiredEventGroupSets");
    cuptiMetricGetAttributePtr              = dlsym(cupti_dlp, "cuptiMetricGetAttribute");
    cuptiMetricGetNumEventsPtr              = dlsym(cupti_dlp, "cuptiMetricGetNumEvents");
    cuptiMetricGetValuePtr                  = dlsym(cupti_dlp, "cuptiMetricGetValue");
    cuptiMetricEnumEventsPtr                = dlsym(cupti_dlp, "cuptiMetricEnumEvents");
    cuptiMetricGetIdFromNamePtr             = dlsym(cupti_dlp, "cuptiMetricGetIdFromName");
    cuptiGetTimestampPtr                    = dlsym(cupti_dlp, "cuptiGetTimestamp");
    cuptiGetVersionPtr                      = dlsym(cupti_dlp, "cuptiGetVersion");
    cuptiSetEventCollectionModePtr          = dlsym(cupti_dlp, "cuptiSetEventCollectionMode");
    cuptiEventDomainEnumEventsPtr           = dlsym(cupti_dlp, "cuptiEventDomainEnumEvents");
    cuptiEventDomainGetAttributePtr         = dlsym(cupti_dlp, "cuptiEventDomainGetAttribute");
    cuptiEventDomainGetNumEventsPtr         = dlsym(cupti_dlp, "cuptiEventDomainGetNumEvents");
    cuptiEventGetAttributePtr               = dlsym(cupti_dlp, "cuptiEventGetAttribute");
    cuptiGetResultStringPtr                 = dlsym(cupti_dlp, "cuptiGetResultString");
    cuptiEnableKernelReplayModePtr          = dlsym(cupti_dlp, "cuptiEnableKernelReplayMode");
    cuptiDisableKernelReplayModePtr         = dlsym(cupti_dlp, "cuptiDisableKernelReplayMode");

    int cupti_not_initialized = (
        !cuptiDeviceEnumMetricsPtr               ||
        !cuptiDeviceGetEventDomainAttributePtr   ||
        !cuptiDeviceGetNumMetricsPtr             ||
        !cuptiDeviceEnumEventDomainsPtr          ||
        !cuptiDeviceGetNumEventDomainsPtr        ||
        !cuptiEventGroupGetAttributePtr          ||
        !cuptiEventGroupReadEventPtr             ||
        !cuptiEventGroupSetAttributePtr          ||
        !cuptiEventGroupSetDisablePtr            ||
        !cuptiEventGroupSetEnablePtr             ||
        !cuptiEventGroupSetsCreatePtr            ||
        !cuptiEventGroupSetsDestroyPtr           ||
        !cuptiEventGroupAddEventPtr              ||
        !cuptiEventGroupCreatePtr                ||
        !cuptiEventGroupDestroyPtr               ||
        !cuptiEventGroupDisablePtr               ||
        !cuptiEventGroupEnablePtr                ||
        !cuptiEventGroupReadAllEventsPtr         ||
        !cuptiEventGroupResetAllEventsPtr        ||
        !cuptiEventGetIdFromNamePtr              ||
        !cuptiMetricCreateEventGroupSetsPtr      ||
        !cuptiMetricGetRequiredEventGroupSetsPtr ||
        !cuptiMetricGetAttributePtr              ||
        !cuptiMetricGetNumEventsPtr              ||
        !cuptiMetricGetValuePtr                  ||
        !cuptiMetricEnumEventsPtr                ||
        !cuptiMetricGetIdFromNamePtr             ||
        !cuptiGetTimestampPtr                    ||
        !cuptiGetVersionPtr                      ||
        !cuptiSetEventCollectionModePtr          ||
        !cuptiEventDomainEnumEventsPtr           ||
        !cuptiEventDomainGetAttributePtr         ||
        !cuptiEventDomainGetNumEventsPtr         ||
        !cuptiEventGetAttributePtr               ||
        !cuptiGetResultStringPtr                 ||
        !cuptiEnableKernelReplayModePtr          ||
        !cuptiDisableKernelReplayModePtr
    );

    papi_errno = (cupti_not_initialized) ? PAPI_EMISC : PAPI_OK;
    if (papi_errno != PAPI_OK) {
        sprintf(error_string, "Error while loading cupti symbols.");
    }

    return papi_errno;
}

int
init_device_table(void)
{
    int papi_errno = PAPI_OK;
    device_t *devices = NULL;

    int count;
    cudaError_t cuda_errno = cudaGetDeviceCountPtr(&count);
    if (cuda_errno != cudaSuccess) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    devices = papi_calloc(count, sizeof(*devices));
    if (devices == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    CUresult cu_errno;
    int i, cc_le7_count = 0, cc_ge7_count = 0;
    for (i = 0; i < count; ++i) {
        cu_errno = cuDeviceGetPtr(&devices[i].cu_device, i);
        if (cu_errno != CUDA_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        devices[i].device_id = i;

        cu_errno = cuDeviceGetNamePtr(devices[i].name, PAPI_MIN_STR_LEN, devices[i].cu_device);
        if (cu_errno != CUDA_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        cuda_errno = cudaDeviceGetAttributePtr(&devices[i].cc_major, cudaDevAttrComputeCapabilityMajor, i);
        if (cuda_errno != cudaSuccess) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        cuda_errno = cudaDeviceGetAttributePtr(&devices[i].cc_minor, cudaDevAttrComputeCapabilityMinor, i);
        if (cuda_errno != cudaSuccess) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        if (devices[i].cc_major < 7 || (devices[i].cc_major == 7 && devices[i].cc_minor == 0)) {
            ++cc_le7_count;
        }

        if (devices[i].cc_major >= 7 && devices[i].cc_minor > 0) {
            ++cc_ge7_count;
        }
    }

    if (cc_le7_count != count) {
        papi_errno = PAPI_ECMP;
        goto fn_fail;
    }

    device_table.count = count;
    device_table.devices = devices;
    device_table_p = &device_table;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (devices) {
        papi_free(devices);
    }
    sprintf(error_string, "Error while initializing the device tables.");
    goto fn_exit;
}

int
init_event_table(void)
{
    int papi_errno = PAPI_OK;
    int i, j, k, count = 0, curr = 0;
    uint32_t num_domains, num_metrics;
    int num_devices = device_table_p->count;
    device_t *devices = device_table_p->devices;
    ntv_event_t *events = NULL;
    CUpti_EventDomainID *cupti_domains_id = NULL;
    CUpti_MetricID *cupti_metrics_id = NULL;
    CUpti_EventID *cupti_events_id = NULL;
    uint64_t *cupti_events_value = NULL;
    size_t size;
    CUptiResult cupti_errno;

    for (i = 0; i < num_devices; ++i) {
        cupti_errno = cuptiDeviceGetNumEventDomainsPtr(devices[i].cu_device, &num_domains);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        cupti_domains_id = papi_realloc(cupti_domains_id, num_domains * sizeof(*cupti_domains_id));
        if (cupti_domains_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        size = num_domains * sizeof(CUpti_EventDomainID);
        cupti_errno = cuptiDeviceEnumEventDomainsPtr(devices[i].cu_device, &size, cupti_domains_id);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        for (j = 0; j < (int) num_domains; ++j) {
            uint32_t num_events;
            cupti_errno = cuptiEventDomainGetNumEventsPtr(cupti_domains_id[j], &num_events);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            cupti_events_id = papi_calloc(num_events, sizeof(*cupti_events_id));
            if (cupti_events_id == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            size = num_events * sizeof(CUpti_EventID);
            cupti_errno = cuptiEventDomainEnumEventsPtr(cupti_domains_id[j], &size, cupti_events_id);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            events = papi_realloc(events, (count + num_events) * sizeof(*events));
            if (events == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            for (k = 0; k < (int) num_events; ++k) {
                events[curr + k].kind = CUPTI_ACTIVITY_KIND_EVENT;

                char name[PAPI_MAX_STR_LEN] = { 0 };
                size = PAPI_MAX_STR_LEN - 1;
                cupti_errno = cuptiEventGetAttributePtr(cupti_events_id[k], CUPTI_EVENT_ATTR_NAME, &size, name);
                if (cupti_errno != CUPTI_SUCCESS) {
                    papi_errno = PAPI_EMISC;
                    goto fn_fail;
                }

                char descr[PAPI_2MAX_STR_LEN] = { 0 };
                size = PAPI_2MAX_STR_LEN - 1;
                cupti_errno = cuptiEventGetAttributePtr(cupti_events_id[k], CUPTI_EVENT_ATTR_SHORT_DESCRIPTION, &size, descr);
                if (cupti_errno != CUPTI_SUCCESS) {
                    papi_errno = PAPI_EMISC;
                    goto fn_fail;
                }

                char full_name[PAPI_2MAX_STR_LEN] = { 0 };
                sprintf(full_name, "event:%s:device=%i", name, i);
                strncpy(events[curr + k].name, full_name, PAPI_MAX_STR_LEN - 1);
                strncpy(events[curr + k].descr, descr, PAPI_2MAX_STR_LEN - 1);
                events[curr + k].u.e.cupti_event_id = cupti_events_id[k];
                events[curr + k].u.e.cupti_event_value = 0;
                events[curr + k].papi_event_id = curr + k;
                events[curr + k].device = &devices[i];
                ++count;
            }
            papi_free(cupti_events_id);
            cupti_events_id = NULL;
            curr += k;
        }

        cupti_errno = cuptiDeviceGetNumMetricsPtr(devices[i].cu_device, &num_metrics);
        if (cupti_errno != PAPI_OK) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        cupti_metrics_id = papi_realloc(cupti_metrics_id, num_metrics * sizeof(*cupti_metrics_id));
        if (cupti_metrics_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        events = papi_realloc(events, (count + num_metrics) * sizeof(*events));
        if (events == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        size = num_metrics * sizeof(CUpti_EventID);
        cupti_errno = cuptiDeviceEnumMetricsPtr(devices[i].cu_device, &size, cupti_metrics_id);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        for (j = 0; j < (int) num_metrics; ++j) {
            events[curr + j].kind = CUPTI_ACTIVITY_KIND_METRIC;

            char name[PAPI_MAX_STR_LEN] = { 0 };
            size = PAPI_MAX_STR_LEN - 1;
            cupti_errno = cuptiMetricGetAttributePtr(cupti_metrics_id[j], CUPTI_METRIC_ATTR_NAME, &size, name);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            char descr[PAPI_2MAX_STR_LEN] = { 0 };
            size = PAPI_2MAX_STR_LEN - 1;
            cupti_errno = cuptiMetricGetAttributePtr(cupti_metrics_id[j], CUPTI_METRIC_ATTR_LONG_DESCRIPTION, &size, descr);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            CUpti_MetricValueKind unit;
            size = sizeof(CUpti_MetricValueKind);
            cupti_errno = cuptiMetricGetAttributePtr(cupti_metrics_id[j], CUPTI_METRIC_ATTR_VALUE_KIND, &size, &unit);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            uint32_t num_cupti_events;
            cupti_errno = cuptiMetricGetNumEventsPtr(cupti_metrics_id[j], &num_cupti_events);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            cupti_events_id = papi_calloc(num_cupti_events, sizeof(*cupti_events_id));
            if (cupti_events_id == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            cupti_events_value = papi_calloc(num_cupti_events, sizeof(*cupti_events_value));
            if (cupti_events_value == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            ++count;

            size = num_cupti_events * sizeof(CUpti_EventID);
            cupti_errno = cuptiMetricEnumEventsPtr(cupti_metrics_id[j], &size, cupti_events_id);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            char full_name[PAPI_2MAX_STR_LEN] = { 0 };
            sprintf(full_name, "metric:%s:device=%i", name, i);
            strncpy(events[curr + j].name, full_name, PAPI_MAX_STR_LEN - 1);
            strncpy(events[curr + j].descr, descr, PAPI_2MAX_STR_LEN - 1);
            events[curr + j].papi_event_id = curr + j;
            events[curr + j].device = &devices[i];
            events[curr + j].u.m.cupti_metric_id = cupti_metrics_id[j];
            events[curr + j].u.m.cupti_events_id = cupti_events_id;
            events[curr + j].u.m.cupti_events_value = cupti_events_value;
            events[curr + j].u.m.num_cupti_events = num_cupti_events;
            events[curr + j].u.m.unit = unit;
        }

        curr += j;
    }

    for (i = 0; i < count; ++i) {
        int htable_errno = htable_insert(htable, events[i].name, &events[i]);
        if (htable_errno != HTABLE_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    ntv_event_table.events = events;
    ntv_event_table.count = count;
    ntv_event_table_p = &ntv_event_table;

  fn_exit:
    papi_free(cupti_domains_id);
    papi_free(cupti_metrics_id);
    return papi_errno;
  fn_fail:
    if (cupti_events_value) {
        papi_free(cupti_events_value);
    }
    if (cupti_events_id) {
        papi_free(cupti_events_id);
    }
    for (i = 0; i < count; ++i) {
        if (events[i].kind == CUPTI_ACTIVITY_KIND_METRIC) {
            if (events[i].u.m.cupti_events_id) {
                papi_free(events[i].u.m.cupti_events_id);
            }
            if (events[i].u.m.cupti_events_value) {
                papi_free(events[i].u.m.cupti_events_value);
            }
        }
    }
    if (events) {
        papi_free(events);
    }
    sprintf(error_string, "Error while initializing the native event table.");
    goto fn_exit;
}

int
unload_cupti_sym(void)
{
    if (cupti_dlp == NULL) {
        return PAPI_OK;
    }

    cuptiDeviceEnumMetricsPtr               = NULL;
    cuptiDeviceGetEventDomainAttributePtr   = NULL;
    cuptiDeviceGetNumMetricsPtr             = NULL;
    cuptiDeviceEnumEventDomainsPtr          = NULL;
    cuptiDeviceGetNumEventDomainsPtr        = NULL;
    cuptiEventGroupGetAttributePtr          = NULL;
    cuptiEventGroupReadEventPtr             = NULL;
    cuptiEventGroupSetAttributePtr          = NULL;
    cuptiEventGroupSetDisablePtr            = NULL;
    cuptiEventGroupSetEnablePtr             = NULL;
    cuptiEventGroupSetsCreatePtr            = NULL;
    cuptiEventGroupSetsDestroyPtr           = NULL;
    cuptiEventGroupAddEventPtr              = NULL;
    cuptiEventGroupCreatePtr                = NULL;
    cuptiEventGroupDestroyPtr               = NULL;
    cuptiEventGroupDisablePtr               = NULL;
    cuptiEventGroupEnablePtr                = NULL;
    cuptiEventGroupReadAllEventsPtr         = NULL;
    cuptiEventGroupResetAllEventsPtr        = NULL;
    cuptiMetricCreateEventGroupSetsPtr      = NULL;
    cuptiMetricGetRequiredEventGroupSetsPtr = NULL;
    cuptiMetricGetAttributePtr              = NULL;
    cuptiMetricGetNumEventsPtr              = NULL;
    cuptiMetricGetValuePtr                  = NULL;
    cuptiMetricEnumEventsPtr                = NULL;
    cuptiGetTimestampPtr                    = NULL;
    cuptiGetVersionPtr                      = NULL;
    cuptiSetEventCollectionModePtr          = NULL;
    cuptiEventDomainEnumEventsPtr           = NULL;
    cuptiEventDomainGetAttributePtr         = NULL;
    cuptiEventDomainGetNumEventsPtr         = NULL;
    cuptiEventGetAttributePtr               = NULL;
    cuptiGetResultStringPtr                 = NULL;
    cuptiEnableKernelReplayModePtr          = NULL;
    cuptiDisableKernelReplayModePtr         = NULL;

    /*
     * cupti_dlp is finalized in cuptic.c
     */
    return PAPI_OK;
}

int
cuptie_shutdown(void)
{
    int papi_errno;

    if (!ntv_event_table_p) {
        return PAPI_OK;
    }

    int htable_errno = htable_shutdown(htable);
    if (htable_errno != HTABLE_SUCCESS) {
        return PAPI_EMISC;
    }

    papi_errno = finalize_event_table();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = finalize_device_table();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return unload_cupti_sym();
}

int
finalize_event_table(void)
{
    int papi_errno = PAPI_OK;

    if (!ntv_event_table_p) {
        return PAPI_OK;
    }

    int i;
    for (i = 0; i < ntv_event_table_p->count; ++i) {
        if (ntv_event_table_p->events[i].kind == CUPTI_ACTIVITY_KIND_METRIC) {
            papi_free(ntv_event_table_p->events[i].u.m.cupti_events_id);
            papi_free(ntv_event_table_p->events[i].u.m.cupti_events_value);
        }
    }
    papi_free(ntv_event_table_p->events);
    ntv_event_table_p->events = NULL;
    ntv_event_table_p->count = 0;
    ntv_event_table_p = NULL;

    return papi_errno;
}

int
finalize_device_table(void)
{
    int papi_errno = PAPI_OK;

    if (!device_table_p) {
        return PAPI_OK;
    }

    papi_free(device_table_p->devices);
    device_table_p->count = 0;
    device_table_p = NULL;

    return papi_errno;
}

static int init_ctl(const char **events, int num_events, cuptic_bitmap_t device_map, cuptic_info_t info, struct cuptie_ctl *ctl);
static int finalize_ctl(struct cuptie_ctl *ctl);
static int create_ctl(struct cuptie_ctl *ctl);
static int destroy_ctl(struct cuptie_ctl *ctl);

/**
 * XXX NOTE: events is an array of event names ordered by device
 */
int
cuptie_ctl_create(const char **events, int num_events, cuptic_info_t info, cuptie_ctl_t *ctl)
{
    int papi_errno;
    struct cuptie_ctl *cupti_ctl = NULL;

    if (num_events < 1) {
        return PAPI_ENOEVNT;
    }

    _papi_hwi_lock(_cuda_lock);

    cuptic_bitmap_t bitmap;
    papi_errno = cuptic_dev_get_map(events, num_events, &bitmap);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptic_dev_acquire(bitmap);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cupti_ctl = papi_calloc(1, sizeof(*cupti_ctl));
    if (cupti_ctl == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = init_ctl(events, num_events, bitmap, info, cupti_ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = create_ctl(cupti_ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cupti_ctl->state |= CUPTIE_CTL_CREATED;
    *ctl = cupti_ctl;

  fn_exit:
    _papi_hwi_unlock(_cuda_lock);
    return papi_errno;
  fn_fail:
    if (cupti_ctl) {
        finalize_ctl(cupti_ctl);
        destroy_ctl(cupti_ctl);
        papi_free(cupti_ctl);
    }
    cuptic_dev_release(bitmap);
    goto fn_exit;
}

int
destroy_ctl(struct cuptie_ctl *ctl)
{
    int papi_errno;
    int num_devices;

    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    CUptiResult cupti_errno;
    int i;
    for (i = 0; i < num_devices; ++i) {
        if (ctl->event_group_sets[i]) {
            cupti_errno = cuptiEventGroupSetsDestroyPtr(ctl->event_group_sets[i]);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }
        }
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

static int get_cupti_events_count(const char **events, int num_events, int *num_cupti_events);

int
init_ctl(const char **events, int num_papi_events, cuptic_bitmap_t device_map, cuptic_info_t info, struct cuptie_ctl *ctl)
{
    int papi_errno = PAPI_OK;
    CUpti_EventID *cupti_events_id = NULL;
    CUpti_EventGroupSets **event_group_sets = NULL;
    uint32_t *papi_events_id = NULL;
    int64_t *counters = NULL;
    int32_t *num_cupti_events_per_dev = NULL;
    int32_t num_devices;

    papi_events_id = papi_calloc(num_papi_events, sizeof(*papi_events_id));
    if (papi_events_id == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    counters = papi_calloc(num_papi_events, sizeof(*counters));
    if (counters == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int num_cupti_events;
    papi_errno = get_cupti_events_count(events, num_papi_events, &num_cupti_events);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cupti_events_id = papi_calloc(num_cupti_events, sizeof(*cupti_events_id));
    if (cupti_events_id == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    papi_errno = cuptic_dev_get_count(device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    num_cupti_events_per_dev = papi_calloc(device_table_p->count, sizeof(*num_cupti_events_per_dev));
    if (num_cupti_events_per_dev == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    event_group_sets = papi_calloc(num_devices, sizeof(*event_group_sets));
    if (event_group_sets == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int i, j, k;
    for (i = 0, j = 0; i < num_papi_events; ++i) {
        ntv_event_t *event;
        if (htable_find(htable, events[i], (void **) &event) != HTABLE_SUCCESS) {
            papi_errno = PAPI_ENOEVNT;
            goto fn_fail;
        }

        papi_events_id[i] = event->papi_event_id;
        if (event->kind == CUPTI_ACTIVITY_KIND_METRIC) {
            cupti_metric_t *metric = &event->u.m;
            for (k = 0; k < metric->num_cupti_events; ++k) {
                cupti_events_id[j++] = metric->cupti_events_id[k];
            }
            num_cupti_events_per_dev[event->device->device_id] += k;
        } else {
            cupti_events_id[j++] = event->u.e.cupti_event_id;
            ++num_cupti_events_per_dev[event->device->device_id];
        }
    }

    ctl->papi_events_id = papi_events_id;
    ctl->num_papi_events = num_papi_events;
    ctl->cupti_events_id = cupti_events_id;
    ctl->num_cupti_events = j;
    ctl->num_cupti_events_per_dev = num_cupti_events_per_dev;
    ctl->device_map = device_map;
    ctl->info = info;
    ctl->counters = counters;
    ctl->event_group_sets = event_group_sets;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
get_cupti_events_count(const char **names, int num_papi_events, int *num_cupti_events)
{
    int papi_errno = PAPI_OK;
    int i;
    *num_cupti_events = 0;

    for (i = 0; i < num_papi_events; ++i) {
        ntv_event_t *event;
        int htable_errno = htable_find(htable, names[i], (void **) &event);
        if (htable_errno != HTABLE_SUCCESS) {
            return PAPI_ECMP;
        }
        if (event->kind == CUPTI_ACTIVITY_KIND_METRIC) {
            *num_cupti_events += event->u.m.num_cupti_events;
        } else {
            ++(*num_cupti_events);
        }
    }

    return papi_errno;
}

int
finalize_ctl(struct cuptie_ctl *ctl)
{
    int papi_errno = PAPI_OK;
    papi_free(ctl->papi_events_id);
    papi_free(ctl->cupti_events_id);
    papi_free(ctl->event_group_sets);
    papi_free(ctl->counters);
    papi_free(ctl->num_cupti_events_per_dev);
    memset(ctl, 0, sizeof(*ctl));
    return papi_errno;
}

int
create_ctl(struct cuptie_ctl *ctl)
{
    int papi_errno;
    int num_devices;
    int device_id;
    uint32_t value = 1;
    int i, j, event_offset = 0;
    size_t size;
    CUptiResult cupti_errno;

    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    for (i = 0; i < num_devices; ++i) {
        papi_errno = cuptic_dev_get_id(ctl->device_map, i, &device_id);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        cupti_errno = cuptiSetEventCollectionModePtr(ctl->info[device_id].ctx, CUPTI_EVENT_COLLECTION_MODE_CONTINUOUS);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        size = ctl->num_cupti_events_per_dev[device_id] * sizeof(CUpti_EventID);
        cupti_errno = cuptiEventGroupSetsCreatePtr(ctl->info[device_id].ctx, size, &ctl->cupti_events_id[event_offset],
                                                   &ctl->event_group_sets[i]);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }

        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[i]->sets[0];
        for (j = 0; j < (int) event_group_set->numEventGroups; ++j) {
            cupti_errno = cuptiEventGroupSetAttributePtr(event_group_set->eventGroups[j],
                                                         CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                                         sizeof(value), &value);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }
        }

        if (ctl->event_group_sets[i] && ctl->event_group_sets[i]->numSets > 1) {
            papi_errno = PAPI_EMULPASS;
            goto fn_fail;
        }

        event_offset += ctl->num_cupti_events_per_dev[device_id];
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptie_ctl_destroy(cuptie_ctl_t ctl)
{
    int papi_errno;

    if (ctl->state != CUPTIE_CTL_CREATED) {
        return PAPI_OK;
    }

    _papi_hwi_lock(_cuda_lock);

    papi_errno = cuptic_dev_release(ctl->device_map);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = destroy_ctl(ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = finalize_ctl(ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_free(ctl);

  fn_exit:
    _papi_hwi_unlock(_cuda_lock);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptie_ctl_start(cuptie_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

    if (ctl->state != CUPTIE_CTL_CREATED) {
        return PAPI_EINVAL;
    }

    int num_devices;
    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    CUptiResult cupti_errno;
    int i, j;
    for (i = 0; i < num_devices; ++i) {
        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[i]->sets[0];
        cupti_errno = cuptiEventGroupSetEnablePtr(event_group_set);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    cupti_errno = cuptiGetTimestampPtr(&ctl->start_time_ns);
    if (cupti_errno != CUPTI_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    ctl->state |= CUPTIE_CTL_STARTED;

  fn_exit:
    return papi_errno;
  fn_fail:
    for (j = 0; j <= i; ++j) {
        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[j]->sets[0];
        cuptiEventGroupSetDisablePtr(event_group_set);
    }
    ctl->start_time_ns = 0;
    goto fn_exit;
}

int
cuptie_ctl_stop(cuptie_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

    if (ctl->state != (CUPTIE_CTL_CREATED | CUPTIE_CTL_STARTED)) {
        return PAPI_EINVAL;
    }

    int num_devices;
    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    CUptiResult cupti_errno;
    int i;
    for (i = 0; i < num_devices; ++i) {
        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[i]->sets[0];
        cupti_errno = cuptiEventGroupSetDisablePtr(event_group_set);
        if (cupti_errno != CUPTI_SUCCESS) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    ctl->state &= ~CUPTIE_CTL_STARTED;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

static int update_event_table_values(struct cuptie_ctl *ctl, int device_id, CUpti_EventID *cupti_events_id, int num_cupti_events,
                                     int total_instances, uint64_t *cupti_events_value);
static int update_ctl_events(struct cuptie_ctl *ctl);

int
cuptie_ctl_read(cuptie_ctl_t ctl, long long **counters)
{
    int papi_errno;
    CUptiResult cupti_errno;
    CUpti_EventID *cupti_events_id = NULL;
    uint64_t *cupti_events_value = NULL;

    if (!(ctl->state & CUPTIE_CTL_STARTED)) {
        return PAPI_EINVAL;
    }

    int num_devices;
    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cupti_errno = cuptiGetTimestampPtr(&ctl->stop_time_ns);
    if (cupti_errno != CUPTI_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    int i, j, k, device_id;
    for (i = 0; i < num_devices; ++i) {
        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[i]->sets[0];

        papi_errno = cuptic_dev_get_id(ctl->device_map, i, &device_id);
        if (papi_errno != PAPI_OK) {
            goto fn_fail;
        }

        CUdevice device = device_table_p->devices[device_id].cu_device;
        for (j = 0; j < (int) event_group_set->numEventGroups; ++j) {
            CUpti_EventGroup event_group = event_group_set->eventGroups[j];
            size_t size = sizeof(CUpti_EventDomainID);
            CUpti_EventDomainID domain_id;
            cupti_errno = cuptiEventGroupGetAttributePtr(event_group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID, &size, &domain_id);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            size = sizeof(uint32_t);
            uint32_t total_instances;
            cupti_errno = cuptiDeviceGetEventDomainAttributePtr(device, domain_id, CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                                                &size, &total_instances);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            size = sizeof(uint32_t);
            uint32_t num_cupti_events;
            cupti_errno = cuptiEventGroupGetAttributePtr(event_group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS, &size, &num_cupti_events);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            cupti_events_id = papi_calloc(num_cupti_events, sizeof(*cupti_events_id));
            if (cupti_events_id == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            size_t events_size = num_cupti_events * sizeof(CUpti_EventID);
            cupti_errno = cuptiEventGroupGetAttributePtr(event_group, CUPTI_EVENT_GROUP_ATTR_EVENTS, &events_size, cupti_events_id);
            if (cupti_errno != CUPTI_SUCCESS) {
                papi_errno = PAPI_EMISC;
                goto fn_fail;
            }

            cupti_events_value = papi_calloc(num_cupti_events * total_instances, sizeof(*cupti_events_value));
            if (cupti_events_value == NULL) {
                papi_errno = PAPI_ENOMEM;
                goto fn_fail;
            }

            for (k = 0; k < (int) num_cupti_events; ++k) {
                size_t value_size = total_instances * sizeof(uint64_t);
                cupti_errno = cuptiEventGroupReadEventPtr(event_group, CUPTI_EVENT_READ_FLAG_NONE, cupti_events_id[k],
                                                          &value_size, cupti_events_value + (k * total_instances));
                if (cupti_errno != CUPTI_SUCCESS) {
                    papi_errno = PAPI_EMISC;
                    goto fn_fail;
                }
            }

            papi_errno = update_event_table_values(ctl, device_id, cupti_events_id, num_cupti_events, total_instances,
                                                   cupti_events_value);
            if (papi_errno != PAPI_OK) {
                goto fn_fail;
            }

            papi_free(cupti_events_id);
            papi_free(cupti_events_value);
        }
    }

    papi_errno = update_ctl_events(ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ctl->start_time_ns = ctl->stop_time_ns;
    *counters = (long long *) ctl->counters;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (cupti_events_id) {
        papi_free(cupti_events_id);
    }
    if (cupti_events_value) {
        papi_free(cupti_events_value);
    }
    goto fn_exit;
}

int
update_event_table_values(struct cuptie_ctl *ctl, int device_id, CUpti_EventID *cupti_events_id, int num_cupti_events,
                          int total_instances, uint64_t *cupti_events_value)
{
    int papi_errno = PAPI_OK;

    ntv_event_t *events = ntv_event_table_p->events;

    uint64_t *events_value = papi_calloc(num_cupti_events, sizeof(uint64_t));
    if (events_value == NULL) {
        return PAPI_ENOMEM;
    }

    int i, j;
    for (i = 0; i < num_cupti_events; ++i) {
        for (j = 0; j < total_instances; ++j) {
            events_value[i] += cupti_events_value[i * total_instances + j];
        }
    }

    int start = 0;
    while (start < ctl->num_papi_events && device_id != events[ctl->papi_events_id[start]].device->device_id) {
        ++start;
    }

    int stop = start;
    while (stop < ctl->num_papi_events && device_id == events[ctl->papi_events_id[stop]].device->device_id) {
        ++stop;
    }

    int k;
    for (i = 0; i < ctl->num_cupti_events; ++i) {
        for (j = start; j < stop; ++j) {
            if (events[ctl->papi_events_id[j]].kind == CUPTI_ACTIVITY_KIND_METRIC) {
                cupti_metric_t *metric = &events[ctl->papi_events_id[j]].u.m;
                for (k = 0; k < metric->num_cupti_events; ++k) {
                    if (cupti_events_id[i] == metric->cupti_events_id[k]) {
                        metric->cupti_events_value[k] = (long long) events_value[i];
                        goto fn_continue;
                    }
                }
            } else {
                cupti_event_t *event = &events[ctl->papi_events_id[j]].u.e;
                if (cupti_events_id[i] == event->cupti_event_id) {
                    event->cupti_event_value = (long long) events_value[i];
                    goto fn_continue;
                }
            }
        }
  fn_continue:;
    }

    papi_free(events_value);

    return papi_errno;
}

int
update_ctl_events(struct cuptie_ctl *ctl)
{
    int papi_errno = PAPI_OK;
    CUptiResult cupti_errno;

    int i;
    ntv_event_t *events = ntv_event_table_p->events;
    for (i = 0; i < ctl->num_papi_events; ++i) {
        ntv_event_t *event = &events[ctl->papi_events_id[i]];
        if (event->kind == CUPTI_ACTIVITY_KIND_METRIC) {
            cupti_metric_t *cupti_metric = &event->u.m;
            CUpti_MetricValue metric_value;
            cupti_errno = cuptiMetricGetValuePtr(event->device->cu_device, cupti_metric->cupti_metric_id,
                                                 cupti_metric->num_cupti_events * sizeof(CUpti_EventID),
                                                 cupti_metric->cupti_events_id,
                                                 cupti_metric->num_cupti_events * sizeof(uint64_t),
                                                 cupti_metric->cupti_events_value,
                                                 ctl->stop_time_ns - ctl->start_time_ns,
                                                 &metric_value);
            if (cupti_errno != CUPTI_SUCCESS) {
                return PAPI_EMISC;
            }

            switch (cupti_metric->unit) {
                case CUPTI_METRIC_VALUE_KIND_DOUBLE:
                    ctl->counters[i] += (long long) metric_value.metricValueDouble;
                    break;
                case CUPTI_METRIC_VALUE_KIND_UINT64:
                    ctl->counters[i] += (long long) metric_value.metricValueUint64;
                    break;
                case CUPTI_METRIC_VALUE_KIND_INT64:
                    ctl->counters[i] += (long long) metric_value.metricValueInt64;
                    break;
                case CUPTI_METRIC_VALUE_KIND_PERCENT:
                    ctl->counters[i] += (long long) metric_value.metricValuePercent;
                    break;
                case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
                    ctl->counters[i] += (long long) metric_value.metricValueThroughput;
                    break;
                case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
                    ctl->counters[i] += (long long) metric_value.metricValueUtilizationLevel;
                    break;
                default:
                    papi_errno = PAPI_EINVAL;
            }
        } else {
            cupti_event_t *cupti_event = &event->u.e;
            ctl->counters[i] += (long long) cupti_event->cupti_event_value;
        }
    }

    return papi_errno;
}

int
cuptie_ctl_reset(cuptie_ctl_t ctl)
{
    int papi_errno = PAPI_OK;

    if (!(ctl->state & CUPTIE_CTL_STARTED)) {
        return PAPI_EINVAL;
    }

    int num_devices;
    papi_errno = cuptic_dev_get_count(ctl->device_map, &num_devices);
    if (papi_errno != PAPI_OK) {
        return PAPI_EINVAL;
    }

    CUptiResult cupti_errno;
    int i;
    for (i = 0; i < num_devices; ++i) {
        CUpti_EventGroupSet *event_group_set = &ctl->event_group_sets[i]->sets[0];
        cupti_errno = cuptiEventGroupResetAllEventsPtr(event_group_set);
        if (cupti_errno != PAPI_OK) {
            papi_errno = PAPI_EMISC;
            goto fn_fail;
        }
    }

    memset(ctl->counters, 0, sizeof(uint64_t) * ctl->num_papi_events);

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
cuptie_evt_enum(unsigned int *event_code, int modifier)
{
    int papi_errno = PAPI_OK;

    switch (modifier) {
        case PAPI_ENUM_FIRST:
            *event_code = 0;
            break;
        case PAPI_ENUM_EVENTS:
            if (*event_code + 1 < (unsigned int) ntv_event_table_p->count) {
                ++(*event_code);
            } else {
                papi_errno = PAPI_END;
            }
            break;
        default:
            papi_errno = PAPI_EINVAL;
    }

    return papi_errno;
}

int
cuptie_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
    int papi_errno = PAPI_OK;

    if (event_code >= (unsigned int) ntv_event_table_p->count) {
        return PAPI_ENOEVNT;
    }
    strncpy(descr, ntv_event_table_p->events[event_code].descr, len);

    return papi_errno;
}

int
cuptie_evt_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = PAPI_OK;

    ntv_event_t *event;
    if (htable_find(htable, name, (void **) &event) != HTABLE_SUCCESS) {
        return PAPI_ECMP;
    }
    *event_code = event->papi_event_id;

    return papi_errno;
}

int
cuptie_evt_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = PAPI_OK;

    if (event_code >= (unsigned int) ntv_event_table_p->count) {
        return PAPI_ENOEVNT;
    }

    ntv_event_t *event = &ntv_event_table_p->events[event_code];
    strncpy(name, event->name, len);

    return papi_errno;
}
