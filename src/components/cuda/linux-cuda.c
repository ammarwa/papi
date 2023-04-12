#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "cuptid.h"
#include "htable.h"

#define PAPI_CUDA_EVENT_MASK (PAPI_NATIVE_MASK >> 1)
#define PAPI_CUDA_CTL_CREATED 0x1
#define PAPI_CUDA_CTL_STARTED 0x2
#define PAPI_CUDA_MAX_COUNTERS 512

papi_vector_t _cuda_vector;
static void *htable;

static int cuda_init_component(int cidx);
static int cuda_shutdown_component(void);
static int cuda_init_thread(hwd_context_t *ctx);
static int cuda_shutdown_thread(hwd_context_t *ctx);

static int cuda_init_control_state(hwd_control_state_t *ctl);
static int cuda_set_domain(hwd_control_state_t * ctrl, int domain);
static int cuda_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx);

static int cuda_cleanup_eventset(hwd_control_state_t *ctl);
static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int cuda_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_init_private(void);

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len);
static int cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

typedef struct {
    int state;
} cuda_context_t;

typedef struct {
    unsigned int *events_id;
    char **events_name;
    int num_events;
    cuptid_info_t cupti_info;
    cuptid_ctl_t cupti_ctl;
} cuda_control_t;

typedef struct {
    unsigned int id;
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_2MAX_STR_LEN];
    int device_id;
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int count;
} ntv_event_table_t;

ntv_event_table_t ntv_event_table;

papi_vector_t _cuda_vector = {
    .cmp_info = {
        .name = "cuda",
        .short_name = "cuda",
        .version = "2.0",
        .description = "CUDA component using the CUPTI API",
        .num_mpx_cntrs = PAPI_CUDA_MAX_COUNTERS,
        .num_cntrs = PAPI_CUDA_MAX_COUNTERS,
        .default_domain = PAPI_DOM_USER,
        .default_granularity = PAPI_GRN_THR,
        .available_granularities = PAPI_GRN_THR,
        .hardware_intr_sig = PAPI_INT_SIGNAL,
        /* component specific cmp_info initializations */
        .fast_real_timer = 0,
        .fast_virtual_timer = 0,
        .attach = 0,
        .attach_must_ptrace = 0,
        .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
        .initialized = 0,
    },
    .size = {
        .context = sizeof(cuda_context_t),
        .control_state = sizeof(cuda_control_t),
    },
    .init_component = cuda_init_component,
    .shutdown_component = cuda_shutdown_component,
    .init_thread = cuda_init_thread,
    .shutdown_thread = cuda_shutdown_thread,

    .ntv_enum_events = cuda_ntv_enum_events,
    .ntv_code_to_name = cuda_ntv_code_to_name,
    .ntv_name_to_code = cuda_ntv_name_to_code,
    .ntv_code_to_descr = cuda_ntv_code_to_descr,
    .ntv_code_to_info = cuda_ntv_code_to_info,

    .init_control_state = cuda_init_control_state,
    .set_domain = cuda_set_domain,
    .update_control_state = cuda_update_control_state,
    .cleanup_eventset = cuda_cleanup_eventset,

    .start = cuda_start,
    .stop = cuda_stop,
    .read = cuda_read,
    .reset = cuda_reset,
};

static int event_table_insert(const char *event_name, unsigned int *event_code);
static int event_table_find(unsigned int event_code, const ntv_event_t **event);
static int check_n_initialize(void);

static int
cuda_init_component(int cidx)
{
    _cuda_vector.cmp_info.CmpIdx = cidx;
    _cuda_vector.cmp_info.num_native_events = -1;
    _cuda_vector.cmp_info.num_cntrs = -1;
    _cuda_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    _cuda_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    sprintf(_cuda_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");

    return PAPI_EDELAY_INIT;
}

static int
cuda_shutdown_component(void)
{
    int papi_errno;

    if (!_cuda_vector.cmp_info.initialized || _cuda_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_OK;
    }

    htable_shutdown(htable);

    papi_errno = cuptid_shutdown();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

  fn_exit:
    _cuda_vector.cmp_info.initialized = 0;
    return papi_errno;
}

static int
cuda_init_private(void)
{
    int papi_errno;

    _papi_hwi_lock(COMPONENT_LOCK);
    if (_cuda_vector.cmp_info.initialized) {
        papi_errno = _cuda_vector.cmp_info.disabled;
        goto fn_exit;
    }

    papi_errno = cuptid_init();
    if (papi_errno != PAPI_OK) {
        const char *error_string;
        cuptid_err_get_last(&error_string);
        strcpy(_cuda_vector.cmp_info.disabled_reason, error_string);
        goto fn_fail;
    }

    int htable_errno = htable_init(&htable);
    if (htable_errno != HTABLE_SUCCESS) {
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

  fn_exit:
    _cuda_vector.cmp_info.initialized = 1;
    _cuda_vector.cmp_info.disabled = papi_errno;
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    cuptid_shutdown();
    goto fn_exit;
}

static int
cuda_init_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int
cuda_shutdown_thread(hwd_context_t __attribute__((unused)) *ctx)
{
    return PAPI_OK;
}

static int
cuda_init_control_state(hwd_control_state_t __attribute__((unused)) *ctl)
{
    return PAPI_OK;
}

static int
cuda_set_domain(hwd_control_state_t __attribute__((unused)) *ctrl, int domain)
{
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain)) {
        return PAPI_OK;
    } else {
        return PAPI_EINVAL;
    }
    return PAPI_OK;
}

static int update_native_events(cuda_control_t *, NativeInfo_t *, int);
static int try_create_cupti_ctl(cuda_control_t *);

static int
cuda_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info, int ntv_count, hwd_context_t *ctx __attribute__((unused)))
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;
    if (cuda_ctl->cupti_ctl != NULL) {
        return PAPI_EINVAL;
    }

    if (ntv_count > 0 && cuda_ctl->cupti_info == CUPTID_INFO_NULL) {
        papi_errno = cuptid_info_create(&cuda_ctl->cupti_info);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
    }

    papi_errno = update_native_events(cuda_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = cuptid_info_update(cuda_ctl->cupti_info, (const char **) cuda_ctl->events_name, cuda_ctl->num_events);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return try_create_cupti_ctl(cuda_ctl);
}

struct event_map_item {
    unsigned int event_id;
    char *event_name;
    int frontend_idx;
    int device_id;
};

static int
compare(const void *a, const void *b)
{
    struct event_map_item *A = (struct event_map_item *) a;
    struct event_map_item *B = (struct event_map_item *) b;
    return A->device_id - B->device_id;
}

int
update_native_events(cuda_control_t *ctl, NativeInfo_t *ntv_info, int ntv_count)
{
    int papi_errno = PAPI_OK;
    int i;

    if (ntv_count != ctl->num_events) {
        ctl->events_id = papi_realloc(ctl->events_id, ntv_count * sizeof(*ctl->events_id));
        if (ctl->events_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        ctl->events_name = papi_realloc(ctl->events_name, ntv_count * sizeof(*ctl->events_name));
        if (ctl->events_name == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }
    }

    struct event_map_item sorted_events[PAPI_CUDA_MAX_COUNTERS] = { 0 };
    for (i = 0; i < ntv_count; ++i) {
        const ntv_event_t *event;
        papi_errno = event_table_find(ntv_info[i].ni_event, &event);
        if (papi_errno == PAPI_OK) {
            sorted_events[i].event_id = ntv_info[i].ni_event;
            sorted_events[i].event_name = papi_strdup(event->name);
            sorted_events[i].device_id = event->device_id;
            sorted_events[i].frontend_idx = i;
        } else if (papi_errno == PAPI_ENOEVNT) {
            char name[PAPI_MAX_STR_LEN] = { 0 };
            papi_errno = cuptid_evt_code_to_name((unsigned int) ntv_info[i].ni_event, name, PAPI_MAX_STR_LEN);
            if (papi_errno != PAPI_OK) {
                goto fn_fail;
            }

            int device_id;
            papi_errno = cuptid_evt_name_to_device(name, &device_id);
            if (papi_errno != PAPI_OK) {
                goto fn_fail;
            }

            sorted_events[i].event_id = ntv_info[i].ni_event;
            sorted_events[i].event_name = papi_strdup(name);
            sorted_events[i].device_id = device_id;
            sorted_events[i].frontend_idx = i;
        } else {
            goto fn_fail;
        }
    }

    qsort(sorted_events, ntv_count, sizeof(struct event_map_item), compare);

    for (i = 0; i < ntv_count; ++i) {
        ctl->events_id[i] = sorted_events[i].event_id;
        ctl->events_name[i] = sorted_events[i].event_name;
        ntv_info[sorted_events[i].frontend_idx].ni_position = i;
    }

    ctl->num_events = ntv_count;

  fn_exit:
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

int
try_create_cupti_ctl(cuda_control_t *ctl)
{
    int papi_errno;

    if (ctl->num_events < 1) {
        return PAPI_OK;
    }

    cuptid_ctl_t cupti_ctl;
    papi_errno = cuptid_ctl_create((const char **) ctl->events_name, ctl->num_events, ctl->cupti_info, &cupti_ctl);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return cuptid_ctl_destroy(cupti_ctl);
}

static int
cuda_cleanup_eventset(hwd_control_state_t *ctl)
{
    int papi_errno = PAPI_OK;

    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctl->cupti_ctl != NULL) {
        return PAPI_EINVAL;
    }

    int i;
    for (i = 0; i < cuda_ctl->num_events; ++i) {
        papi_free(cuda_ctl->events_name[i]);
    }

    papi_free(cuda_ctl->events_id);
    papi_free(cuda_ctl->events_name);
    cuda_ctl->num_events = 0;

    papi_errno = cuptid_info_destroy(cuda_ctl->cupti_info);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuda_ctl->cupti_info = CUPTID_INFO_NULL;

    return papi_errno;
}

static int
cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno;

    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctx->state & (PAPI_CUDA_CTL_CREATED | PAPI_CUDA_CTL_STARTED)) {
        return PAPI_EINVAL;
    }

    papi_errno = cuptid_ctl_create((const char **) cuda_ctl->events_name, cuda_ctl->num_events, cuda_ctl->cupti_info,
                                   &cuda_ctl->cupti_ctl);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuda_ctx->state = PAPI_CUDA_CTL_CREATED;

    papi_errno = cuptid_ctl_start(cuda_ctl->cupti_ctl);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    cuda_ctx->state |= PAPI_CUDA_CTL_STARTED;

  fn_exit:
    return papi_errno;
  fn_fail:
    cuptid_ctl_destroy(cuda_ctl->cupti_ctl);
    cuda_ctx->state = 0;
    goto fn_exit;
}

int
cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    int papi_errno;

    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctx->state != (PAPI_CUDA_CTL_CREATED | PAPI_CUDA_CTL_STARTED)) {
        return PAPI_EINVAL;
    }

    papi_errno = cuptid_ctl_stop(cuda_ctl->cupti_ctl);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    cuda_ctx->state &= ~PAPI_CUDA_CTL_STARTED;

    papi_errno = cuptid_ctl_destroy(cuda_ctl->cupti_ctl);

    cuda_ctx->state = 0;
    cuda_ctl->cupti_ctl = NULL;

    return papi_errno;
}

static int
cuda_read(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl, long long **val, int __attribute__((unused)) flags)
{
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctl->cupti_ctl == NULL) {
        return PAPI_EINVAL;
    }

    return cuptid_ctl_read(cuda_ctl->cupti_ctl, val);
}

static int
cuda_reset(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t __attribute__((unused)) *ctl)
{
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctl->cupti_ctl == NULL) {
        return PAPI_EINVAL;
    }

    return cuptid_ctl_reset(cuda_ctl->cupti_ctl);
}

static int
evt_get_count(int *count)
{
    unsigned int event_code = 0;

    if (cuptid_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
    }
    while (cuptid_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}

static int
cuda_ntv_enum_events(unsigned int *event_code, int modifier)
{

    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    if (ntv_event_table.count > 0) {
        switch(modifier) {
            case PAPI_ENUM_FIRST:
                *event_code = 0 | PAPI_CUDA_EVENT_MASK;
                return PAPI_OK;
            case PAPI_ENUM_EVENTS:
                if (*event_code + 1 < (unsigned int) ntv_event_table.count) {
                    *event_code = (*event_code + 1) | PAPI_CUDA_EVENT_MASK;
                    return PAPI_OK;
                }
                break;
            default:
                return PAPI_EINVAL;
        }
    }

    papi_errno = cuptid_evt_enum(event_code, modifier);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    static int count;
    if (count == 0) {
        papi_errno = evt_get_count(&count);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        _cuda_vector.cmp_info.num_native_events = count;
        _cuda_vector.cmp_info.num_cntrs = count;
    }

    return papi_errno;
}

static int
cuda_ntv_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    ntv_event_t *event;
    int htable_errno = htable_find(htable, name, (void **) &event);
    if (htable_errno == HTABLE_SUCCESS) {
        *event_code = event->id;
    } else if (htable_errno == HTABLE_ENOVAL) {
        papi_errno = cuptid_evt_name_to_code(name, event_code);
        if (papi_errno == PAPI_EDELAY_INIT) {
            papi_errno = event_table_insert(name, event_code);
        }
    } else {
        papi_errno = PAPI_ECMP;
    }

    return papi_errno;
}

int
event_table_insert(const char *event_name, unsigned int *event_code)
{
    int papi_errno = PAPI_OK;

    ntv_event_t *events = ntv_event_table.events;
    int num_events = ++ntv_event_table.count;

    events = papi_realloc(events, num_events * sizeof(ntv_event_t));
    if (events == NULL) {
        papi_errno = PAPI_ENOMEM;
        goto fn_fail;
    }

    int device_id;
    papi_errno = cuptid_evt_name_to_device(event_name, &device_id);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    ntv_event_t *event = &events[num_events - 1];
    event->id = (num_events - 1) | PAPI_CUDA_EVENT_MASK;
    strcpy(event->name, event_name);
    strcpy(event->descr, "");
    event->device_id = device_id;

    *event_code = event->id;
    ntv_event_table.events = events;

  fn_exit:
    return papi_errno;
  fn_fail:
    if (events) {
        papi_free(events);
    }
    goto fn_exit;
}

static int
cuda_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    const ntv_event_t *event;
    papi_errno = event_table_find(event_code, &event);
    if (papi_errno == PAPI_OK) {
        strncpy(name, event->name, len);
    } else if (papi_errno == PAPI_ENOEVNT) {
        papi_errno = cuptid_evt_code_to_name(event_code, name, len);
        if (papi_errno == PAPI_EDELAY_INIT) {
            papi_errno = PAPI_ENOEVNT;
        }
    }

    return papi_errno;
}

int
event_table_find(unsigned int event_code, const ntv_event_t **event)
{
    if (!(event_code & PAPI_CUDA_EVENT_MASK)) {
        return PAPI_ENOEVNT;
    }

    int event_idx = event_code & ~PAPI_CUDA_EVENT_MASK;
    if (event_idx >= ntv_event_table.count) {
        return PAPI_ENOEVNT;
    }

    *event = &ntv_event_table.events[event_idx];
    return PAPI_OK;
}

static int
cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int __attribute__((unused)) len)
{
    int papi_errno;

    papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    ntv_event_t *event;
    papi_errno = event_table_find(event_code, (const ntv_event_t **) &event);
    if (papi_errno == PAPI_OK) {
        if (strlen(event->descr) == 0) {
            unsigned int code;
            /* Call cuptid_evt_enum to initialize event table in the backend */
            papi_errno = cuptid_evt_enum(&code, PAPI_ENUM_FIRST);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            papi_errno = cuptid_evt_name_to_code(event->name, &code);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            char descriptor[PAPI_2MAX_STR_LEN] = { 0 };
            papi_errno = cuptid_evt_code_to_descr(code, descriptor, PAPI_2MAX_STR_LEN);
            if (papi_errno != PAPI_OK) {
                return papi_errno;
            }

            /* update event descriptor in frontend event table */
            strncpy(event->descr, descriptor, PAPI_2MAX_STR_LEN);
        }
        strncpy(descr, event->descr, PAPI_2MAX_STR_LEN);
    } else if (papi_errno == PAPI_ENOEVNT) {
        char descriptor[PAPI_2MAX_STR_LEN] = { 0 };
        papi_errno = cuptid_evt_code_to_descr(event_code, descriptor, PAPI_2MAX_STR_LEN);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        strncpy(descr, descriptor, PAPI_2MAX_STR_LEN);
    }

    return papi_errno;
}

int
cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    ntv_event_t *event;
    int papi_errno = event_table_find(event_code, (const ntv_event_t **) &event);
    if (papi_errno == PAPI_OK) {
        strcpy(info->symbol, event->name);
        strcpy(info->long_descr, event->descr);
    } else if (papi_errno == PAPI_ENOEVNT) {
        papi_errno = cuptid_evt_code_to_name(event_code, info->symbol, PAPI_MAX_STR_LEN);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }
        papi_errno = cuptid_evt_code_to_descr(event_code, info->long_descr, PAPI_2MAX_STR_LEN);
    }
    return papi_errno;
}

int
check_n_initialize(void)
{
    if (!_cuda_vector.cmp_info.initialized) {
        return cuda_init_private();
    }
    return _cuda_vector.cmp_info.disabled;
}
