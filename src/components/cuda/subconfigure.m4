[#] start of __file__

AC_ARG_WITH([cuda-inc],
            [AS_HELP_STRING([--with-cuda-inc=@<:@PATH@:>@],
                            [Path to the cuda runtime include header. Default if not given is $PAPI_CUDA_ROOT/include.])],
            [
                WITH_CUDA_INC=$with_cuda_inc
                HAVE_CUDA_INC=1
            ],
            [])

AC_ARG_WITH([cuda-lib],
            [AS_HELP_STRING([--with-cuda-lib=@<:@PATH@:>@],
                            [Path to the cuda runtime library. Default if not given is $PAPI_CUDA_ROOT/lib64.])],
            [
                WITH_CUDA_LIB=$with_cuda_lib
                HAVE_CUDA_LIB=1
            ],
            [])

AC_ARG_WITH([cuda],
            [AS_HELP_STRING([--with-cuda=@<:@PATH@:>@],
                            [Path to the cuda runtime library installation. Default if not given is $PAPI_CUDA_ROOT.])],
            [
                with_cuda=$with_cuda
                WITH_CUDA_INC=$with_cuda/include
                WITH_CUDA_LIB=$with_cuda/lib64
                HAVE_CUDA_INC=1
                HAVE_CUDA_LIB=1
            ],
            [])
AC_SUBST([WITH_CUDA_INC])
AC_SUBST([WITH_CUDA_LIB])
AC_SUBST([HAVE_CUDA_INC])
AC_SUBST([HAVE_CUDA_LIB])

AC_ARG_WITH([cupti-inc],
            [AS_HELP_STRING([--with-cupti-inc=@<:@PATH@:>@],
                            [Path to the cupti profiling include header. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI/include.])],
            [
                WITH_CUPTI_INC=$with_cupti_inc
                HAVE_CUPTI_INC=1
            ],
            [])

AC_ARG_WITH([cupti-lib],
            [AS_HELP_STRING([--with-cupti-lib=@<:@PATH@:>@],
                            [Path to cupti profiling library. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI/lib64.])],
            [
                WITH_CUPTI_LIB=$with_cupti_lib
                HAVE_CUPTI_LIB=1
            ],
            [])

AC_ARG_WITH([cupti],
            [AS_HELP_STRING([--with-cupti=@<:@PATH@:>@],
                            [Path to the cupti profiling library. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI.])],
            [
                with_cupti=$with_cupti
                WITH_CUPTI_INC=$with_cupti/include
                WITH_CUPTI_LIB=$with_cupti/lib64
                HAVE_CUPTI_INC=1
                HAVE_CUPTI_LIB=1
            ],
            [])
AC_SUBST([WITH_CUPTI_INC])
AC_SUBST([WITH_CUPTI_LIB])
AC_SUBST([HAVE_CUPTI_INC])
AC_SUBST([HAVE_CUPTI_LIB])

AC_ARG_WITH([perfworks-inc],
            [AS_HELP_STRING([--with-perfworks-inc=@<:@PATH@:>@],
                            [Path to perfworks include header. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI/include.])],
            [
                WITH_PERFWORKS_INC=$with_perfworks_inc
                HAVE_PERFWORKS_INC=1
            ],
            [])

AC_ARG_WITH([perfworks-lib],
            [AS_HELP_STRING([--with-perfoworks-lib=@<:@PATH@:>@],
                            [Path to perfworks library. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI/lib64.])],
            [
                WITH_PERFWORKS_LIB=$with_perfworks_lib
                HAVE_PERFWORKS_LIB=1
            ],
            [])

AC_ARG_WITH([perfworks],
            [AS_HELP_STRING([--with-perfworks=@<:@PATH@:>@],
                            [Path to perfworks library. Default if not given is $PAPI_CUDA_ROOT/extras/CUPTI/lib64.])],
            [
                with_perfworks=$with_perfworks
                WITH_PERFWORKS_INC=$with_perfworks/include
                WITH_PERFWORKS_LIB=$with_perfworks/lib64
                HAVE_PERFWORKS_INC=1
                HAVE_PERFWORKS_LIB=1
            ],
            [])
AC_SUBST([WITH_PERFWORKS_INC])
AC_SUBST([WITH_PERFWORKS_LIB])
AC_SUBST([HAVE_PERFWORKS_INC])
AC_SUBST([HAVE_PERFWORKS_LIB])

[#] end of __file__
