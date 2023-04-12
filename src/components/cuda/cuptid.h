/**
 * This is the main interface the component frontend interacts with
 * to access cupti. It provides a dispatching (cuptid_) layer to the
 * lower level cupti profiler and event APIs. The cuptid layer
 * dispatches component frontend calls to the appropriate cupti
 * backend API depending on system and user configuration.
 */
#ifndef __CUPTID_H__
#define __CUPTID_H__

#define CUPTID_INFO_NULL 0

extern unsigned int _cuda_lock;

typedef void *cuptid_ctl_t;
typedef void *cuptid_info_t;

int cuptid_init(void);
int cuptid_shutdown(void);

int cuptid_info_create(cuptid_info_t *info);
int cuptid_info_update(cuptid_info_t info, const char **events, int num_events);
int cuptid_info_destroy(cuptid_info_t info);

int cuptid_err_get_last(const char **err_string);

int cuptid_ctl_create(const char **events, int num_events, cuptid_info_t info, cuptid_ctl_t *ctl);
int cuptid_ctl_destroy(cuptid_ctl_t ctl);
int cuptid_ctl_start(cuptid_ctl_t ctl);
int cuptid_ctl_stop(cuptid_ctl_t ctl);
int cuptid_ctl_read(cuptid_ctl_t ctl, long long **counters);
int cuptid_ctl_reset(cuptid_ctl_t ctl);

int cuptid_evt_enum(unsigned int *event_code, int modifier);
int cuptid_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int cuptid_evt_name_to_code(const char *name, unsigned int *event_code);
int cuptid_evt_name_to_device(const char *name, int *device_id);
int cuptid_evt_code_to_name(unsigned int event_code, char *name, int len);

#endif /* End of__CUPTID_H__ */
