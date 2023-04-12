#include <papi.h>
#include "cuptid.h"
#include "cuptic.h"

#if defined(API_PERFWORKS)
#include "cuptip.h"
#endif

#if defined(API_EVENTS)
#include "cuptie.h"
#endif

int
cuptid_init(void)
{
    int papi_errno = cuptic_init();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

#if defined(API_PERFWORKS)
    papi_errno = (!(cuptic_api_fallback_event())) ?
        cuptip_init() :
        cuptie_init();
#else
    papi_errno = cuptie_init();
#endif

    return papi_errno;
}

int
cuptid_shutdown(void)
{
    int papi_errno;

#if defined(API_PERFWORKS)
    papi_errno = (!(cuptic_api_fallback_event())) ?
        cuptip_shutdown() :
        cuptie_shutdown();
#else
    papi_errno = cuptie_shutdown();
#endif

    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    return cuptic_shutdown();
}

int
cuptid_info_create(cuptid_info_t *info)
{
    return cuptic_info_create((cuptic_info_t *) info);
}

int
cuptid_info_update(cuptid_info_t info, const char **events, int num_events)
{
    return cuptic_info_update((cuptic_info_t) info, events, num_events);
}

int
cuptid_info_destroy(cuptid_info_t info)
{
    return cuptic_info_destroy((cuptic_info_t) info);
}

int
cuptid_err_get_last(const char **err_string)
{
    return cuptic_err_get_last(err_string);
}

int
cuptid_ctl_create(const char **events, int num_events, cuptid_info_t info, cuptid_ctl_t *ctl)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_create(events, num_events, (cuptic_info_t) info, (cuptip_ctl_t *) ctl);
    }
    return cuptie_ctl_create(events, num_events, (cuptic_info_t) info, (cuptie_ctl_t *) ctl);
#endif
    return cuptie_ctl_create(events, num_events, (cuptic_info_t) info, (cuptie_ctl_t *) ctl);
}

int
cuptid_ctl_destroy(cuptid_ctl_t ctl)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_destroy((cuptip_ctl_t) ctl);
    }
    return cuptie_ctl_destroy((cuptie_ctl_t) ctl);
#endif
    return cuptie_ctl_destroy((cuptie_ctl_t) ctl);
}

int
cuptid_ctl_start(cuptid_ctl_t ctl)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_start((cuptip_ctl_t) ctl);
    }
    return cuptie_ctl_start((cuptie_ctl_t) ctl);
#endif
    return cuptie_ctl_start((cuptie_ctl_t) ctl);
}

int
cuptid_ctl_stop(cuptid_ctl_t ctl)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_stop((cuptip_ctl_t) ctl);
    }
    return cuptie_ctl_stop((cuptie_ctl_t) ctl);
#endif
    return cuptie_ctl_stop((cuptie_ctl_t) ctl);
}

int
cuptid_ctl_read(cuptid_ctl_t ctl, long long **counters)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_read((cuptip_ctl_t) ctl, counters);
    }
    return cuptie_ctl_read((cuptie_ctl_t) ctl, counters);
#endif
    return cuptie_ctl_read((cuptie_ctl_t) ctl, counters);
}

int
cuptid_ctl_reset(cuptid_ctl_t ctl)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_ctl_reset((cuptip_ctl_t) ctl);
    }
    return cuptie_ctl_reset((cuptie_ctl_t) ctl);
#endif
    return cuptie_ctl_reset((cuptie_ctl_t) ctl);
}

int
cuptid_evt_enum(unsigned int *event_code, int modifier)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_evt_enum(event_code, modifier);
    }
    return cuptie_evt_enum(event_code, modifier);
#endif
    return cuptie_evt_enum(event_code, modifier);
}

int
cuptid_evt_code_to_descr(unsigned int event_code, char *descr, int len)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_evt_code_to_descr(event_code, descr, len);
    }
    return cuptie_evt_code_to_descr(event_code, descr, len);
#endif
    return cuptie_evt_code_to_descr(event_code, descr, len);
}

int
cuptid_evt_name_to_code(const char *name, unsigned int *event_code)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_evt_name_to_code(name, event_code);
    }
    return cuptie_evt_name_to_code(name, event_code);
#endif
    return cuptie_evt_name_to_code(name, event_code);
}

int
cuptid_evt_name_to_device(const char *name, int *device_id)
{
    return cuptic_evt_get_device(name, device_id);
}

int
cuptid_evt_code_to_name(unsigned int event_code, char *name, int len)
{
#if defined(API_PERFWORKS)
    if (!(cuptic_api_fallback_event())) {
        return cuptip_evt_code_to_name(event_code, name, len);
    }
    return cuptie_evt_code_to_name(event_code, name, len);
#endif
    return cuptie_evt_code_to_name(event_code, name, len);
}
