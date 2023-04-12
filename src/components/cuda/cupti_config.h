/**
 * Include appropriate CUDA APIs. Set flags to check which API.
 */
#ifndef __CUPTI_CONFIG_H__
#define __CUPTI_CONFIG_H__

#include <cuda.h>
#include <cupti.h>

#if CUPTI_API_VERSION >= 13
#define API_PERFWORKS 1
#endif

#if CUPTI_API_VERSION < 17
#define API_EVENTS 1
#endif

#endif  // __CUPTI_CONFIG_H__
