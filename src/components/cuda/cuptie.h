/**
 * This is the cupti event and metrics (cuptie_) wrapper interface.
 */
#ifndef __CUPTIE_H__
#define __CUPTIE_H__

#include "cuptic.h"

typedef struct cuptie_ctl *cuptie_ctl_t;

int cuptie_init(void);
int cuptie_shutdown(void);

int cuptie_ctl_create(const char **events, int num_events, cuptic_info_t info, cuptie_ctl_t *ctl);
int cuptie_ctl_destroy(cuptie_ctl_t ctl);
int cuptie_ctl_start(cuptie_ctl_t ctl);
int cuptie_ctl_stop(cuptie_ctl_t ctl);
int cuptie_ctl_read(cuptie_ctl_t ctl, long long **counters);
int cuptie_ctl_reset(cuptie_ctl_t ctl);

int cuptie_evt_enum(unsigned int *event_code, int modifier);
int cuptie_evt_code_to_descr(unsigned int event_code, char *descr, int len);
int cuptie_evt_name_to_code(const char *name, unsigned int *event_code);
int cuptie_evt_code_to_name(unsigned int event_code, char *name, int len);

#endif  /* End of __CUPTIE_H__ */
