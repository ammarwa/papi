/**
 * This is the cupti profiler (cuptip_) wrapper interface
 */
#ifndef __CUPTIP_H__
#define __CUPTIP_H__

#include "cuptic.h"

typedef struct cuptip_ctl *cuptip_ctl_t;

int cuptip_init(void);
int cuptip_shutdown(void);

int cuptip_ctl_create(const char **events, int num_events, cuptic_info_t info, cuptip_ctl_t *ctl);
int cuptip_ctl_destroy(cuptip_ctl_t ctl);
int cuptip_ctl_start(cuptip_ctl_t ctl);
int cuptip_ctl_stop(cuptip_ctl_t ctl);
int cuptip_ctl_read(cuptip_ctl_t ctl, long long **counters);
int cuptip_ctl_reset(cuptip_ctl_t ctl);

int cuptip_evt_enum(unsigned int *event_code, int modifier);
int cuptip_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int cuptip_evt_name_to_code(const char *name, unsigned int *event_code);
int cuptip_evt_code_to_name(unsigned int event_code, char *name, int len);

#endif  /* End of __CUPTIP_H__ */
