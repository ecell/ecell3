dnl $Id$ -*- autoconf -*-
dnl
dnl
AC_DEFUN([_SUBST_DEFINE],
[
$1=$2
AC_SUBST($1)
AC_DEFINE_UNQUOTED($1,"$2")
])



## this one is commonly used with AM_PATH_PYTHONDIR ...
dnl AM_CHECK_PYMOD(MODNAME [,SYMBOL [,ACTION-IF-FOUND [,ACTION-IF-NOT-FOUND]]])
dnl Check if a module containing a given symbol is visible to python.
AC_DEFUN([AM_CHECK_PYMOD],
[AC_REQUIRE([AM_PATH_PYTHON])
py_mod_var=`echo $1['_']$2 | sed 'y%./+-%__p_%'`
AC_MSG_CHECKING(for ifelse([$2],[],,[$2 in ])python module $1)
AC_CACHE_VAL(py_cv_mod_$py_mod_var, [
ifelse([$2],[], [prog="
import sys
try:
        import $1
except ImportError:
        sys.exit(1)
except:
        sys.exit(0)
sys.exit(0)"], [prog="
import $1
$1.$2"])
if $PYTHON -c "$prog" 1>&AC_FD_CC 2>&AC_FD_CC
  then
    eval "py_cv_mod_$py_mod_var=yes"
  else
    eval "py_cv_mod_$py_mod_var=no"
  fi
])
py_val=`eval "echo \`echo '$py_cv_mod_'$py_mod_var\`"`
if test "x$py_val" != xno; then
  AC_MSG_RESULT(yes)
  ifelse([$3], [],, [$3
])dnl
else
  AC_MSG_RESULT(no)
  ifelse([$4], [],, [$4
])dnl
fi
])

dnl a macro to check for ability to create python extensions
dnl  AM_CHECK_PYTHON_HEADERS([ACTION-IF-POSSIBLE], [ACTION-IF-NOT-POSSIBLE])
dnl function also defines PYTHON_INCLUDES
AC_DEFUN([AM_CHECK_PYTHON_HEADERS],
[AC_REQUIRE([AM_PATH_PYTHON])
AC_MSG_CHECKING(for headers required to compile python extensions)
dnl deduce PYTHON_INCLUDES
py_prefix=`$PYTHON -c "import sys; print sys.prefix"`
py_exec_prefix=`$PYTHON -c "import sys; print sys.exec_prefix"`
if test -x "$PYTHON_PREFIX/bin/python-config"; then
  PYTHON_INCLUDES=`$PYTHON_PREFIX/bin/python-config --includes`
else
  PYTHON_INCLUDES="$PYTHON_INCLUDES -I${py_exec_prefix}/include/python${PYTHON_VERSION}"
fi
AC_SUBST(PYTHON_INCLUDES)
dnl check if the headers exist:
save_CPPFLAGS="$CPPFLAGS"
CPPFLAGS="$CPPFLAGS $PYTHON_INCLUDES"
AC_TRY_CPP([#include <Python.h>],dnl
[AC_MSG_RESULT(found)
$1],dnl
[AC_MSG_RESULT(not found)
$2])
CPPFLAGS="$save_CPPFLAGS"
])


AC_DEFUN([ECELL_CHECK_PYTHON_LIBS], [
  AC_REQUIRE([AM_PATH_PYTHON])
  AC_MSG_CHECKING([for python library location])
  PYTHON_LIBNAME=`$PYTHON -c 'import sys; print "python%d.%d" % (sys.hexversion >> 24, (sys.hexversion >> 16) & 0xff)'`
  PYTHON_PREFIX=`$PYTHON -c 'import sys; print sys.prefix'`
  if test -x "$PYTHON_PREFIX/bin/python-config"; then
    PYTHON_LIBS="-L$PYTHON_PREFIX/lib `$PYTHON_PREFIX/bin/python-config --ldflags`"
  else
    PYTHON_LIBS="-L$PYTHON_PREFIX/lib -l$PYTHON_LIBNAME"
  fi

  ac_save_LIBS="$LIBS"
  ac_save_CPPFLAGS="$CPPFLAGS"
  LIBS="$PYTHON_LIBS"
  CPPFLAGS="$PYTHON_INCLUDES"
  AC_TRY_LINK([
#include <Python.h>
  ], [
Py_Initialize();
  ],, [
    PYTHON_LIBS=
    case $host_os in
    darwin* | rhapsody*)
      LIBS="-F$PYTHON_PREFIX/../.."
      AC_TRY_LINK([
#include <python/Python.h>
      ], [
Py_Initialize();
      ], [
        PYTHON_LIBS="$LIBS -framework Python"
      ])
      ;;
    esac
  ])
  LIBS="$ac_save_LIBS"
  CPPFLAGS="$ac_save_CPPFLAGS"

  if test -z "$PYTHON_LIBS"; then
    AC_MSG_RESULT([no])
  else
    AC_MSG_RESULT([$PYTHON_LIBS])
  fi
])


AC_DEFUN([ECELL_CHECK_MATH_HEADER], [
  STD_MATH_HEADER=

  CANDIDATES="math cmath math.h"
  for hdr in $CANDIDATES; do
    if test -z "$STD_MATH_HEADER"; then
      AC_CHECK_HEADER([$hdr], [STD_MATH_HEADER="<$hdr>"])
    fi
  done

  if test -z "$STD_MATH_HEADER"; then
    AC_MSG_ERROR([No usable math headers ($CANDIDATES) found])
  fi

  AC_SUBST([STD_MATH_HEADER])
])


AC_DEFUN([ECELL_CHECK_INFINITY], [
  AC_MSG_CHECKING(for INFINITY)
  AC_CACHE_VAL([ac_cv_func_or_macro_infinity], [
    AC_TRY_COMPILE(
    [#include $STD_MATH_HEADER], [
      double inf = INFINITY;
    ], [
      ac_cv_func_or_macro_infinity=yes
    ], [
      ac_cv_func_or_macro_infinity=no
    ])
  ])
  if test "$ac_cv_func_or_macro_infinity" = "yes"; then
    AC_DEFINE(HAVE_INFINITY, 1, [Define to 1 if INFINITY is available.])
    AC_MSG_RESULT(yes)
    HAVE_INFINITY=1
  else
    AC_MSG_RESULT(no)
    HAVE_INFINITY=
  fi
])


AC_DEFUN([ECELL_CHECK_HUGE_VAL], [
  AC_MSG_CHECKING(for HUGE_VAL)
  AC_CACHE_VAL([ac_cv_func_or_macro_huge_val], [
    AC_TRY_COMPILE(
    [#include $STD_MATH_HEADER], [
      double inf = HUGE_VAL;
    ], [
      ac_cv_func_or_macro_huge_val=yes
    ], [
      ac_cv_func_or_macro_huge_val=no
    ])
  ])
  if test "$ac_cv_func_or_macro_huge_val" = "yes"; then
    AC_DEFINE(HAVE_HUGE_VAL, 1, [Define to 1 if HUGE_VAL is available.])
    AC_MSG_RESULT(yes)
    HAVE_HUGE_VAL=1
  else
    AC_MSG_RESULT(no)
    HAVE_HUGE_VAL=
  fi
])


AC_DEFUN([ECELL_CHECK_NUMERIC_LIMITS_DOUBLE_INFINITY], [
  AC_MSG_CHECKING(for numeric_limits<double>::infinity())
  AC_CACHE_VAL([ac_cv_func_or_macro_numeric_limits_double_infinity], [
    AC_TRY_COMPILE(
    [#include <limits>], [
      double inf(std::numeric_limits<double>::infinity());
    ], [
      ac_cv_func_or_macro_numeric_limits_double_infinity=yes
    ], [
      ac_cv_func_or_macro_numeric_limits_double_infinity=no
    ])
  ])
  if test "$ac_cv_func_or_macro_numeric_limits_double_infinity" = "yes"; then
    AC_DEFINE(HAVE_NUMERIC_LIMITS_DOUBLE_INFINTIY, 1, [Define to 1 if std::numeric_limits<double>::infinity() is available.])
    AC_MSG_RESULT(yes)
    HAVE_NUMERIC_LIMITS_DOUBLE_INFINITY=1
  else
    AC_MSG_RESULT(no)
    HAVE_NUMERIC_LIMITS_DOUBLE_INFINITY=
  fi
])


AC_DEFUN([ECELL_CHECK_PRETTY_FUNCTION], [
  AC_MSG_CHECKING(for __PRETTY_FUNCTION__)
  AC_CACHE_VAL(ac_cv_macro_pretty_function, [
    AC_TRY_CPP([
  test()
  {
    const char* pretty_function = __PRETTY_FUNCTION__;
  }
    ], [
      ac_cv_macro_pretty_function=yes
    ], [
      ac_cv_macro_pretty_function=no
    ])
  ])
  if [[ $ac_cv_macro_pretty_function == yes ]]; then
    AC_DEFINE(HAVE_PRETTY_FUNCTION, 1, dnl
      [Define to 1 if __PRETTY_FUNCTION__ is supported.])
    AC_MSG_RESULT(yes)
  else
    AC_MSG_RESULT(no)
  fi
])


AC_DEFUN([ECELL_CHECK_GSL], [
  GSL_CONFIG=
  AC_CHECK_PROGS([GSL_CONFIG], [gsl-config])

  if test -z "$GSL_CONFIG"; then
    AC_CHECK_LIB([gslcblas], [cblas_dgemm],, [
      AC_MSG_ERROR([libgslcblas could not be found.])
    ])

    AC_CHECK_LIB([gsl], [gsl_block_alloc],, [
      AC_MSG_ERROR([libgsl could not be found.])
    ])
  else
    CPPFLAGS="$CPPFLAGS `$GSL_CONFIG --cflags`"
    LIBS="$LIBS `$GSL_CONFIG --libs`"
  fi

  # DLL import spec would be necessary under Win32.
  case ${host_os} in
    *cygwin* | *mingw* | *win32*)
      AC_MSG_CHECKING([if link is performed against dynamic GSL libraries])
      ac_save_LDFLAGS="$LDFLAGS"
      LDFLAGS="-Wl,-Bdynamic $LDFLAGS"
      AC_TRY_LINK([
#include <gsl/gsl_vector.h>
      ], [
gsl_vector_alloc(0);
      ], [
        AC_MSG_RESULT([yes])
        CPPFLAGS="$CPPFLAGS -DGSL_DLL" 
      ], [
        AC_MSG_RESULT([no])
      ])
      LDFLAGS="$ac_save_LDFLAGS"
      ;;
  esac
])

AC_DEFUN([ECELL_CHECK_BOOST], [
  AC_CHECK_HEADER([boost/config.hpp], [
    AC_DEFINE(HAVE_BOOST, 1, [Define to 1 if Boost C++ library is installed])
    ifelse([$1], [],, [$1])
  ], [
    ifelse([$2], [],, [$2])
  ])
])

AC_DEFUN([ECELL_CHECK_BOOST_PYTHON], [
  AC_REQUIRE([AM_CHECK_PYTHON_HEADERS])

  BOOST_PYTHON_LIBNAME=boost_python
  AC_ARG_WITH(boost-python-libname,
    AC_HELP_STRING([--with-boost-python-libname=LIBNAME],
                   [specify the library name of Boost.Python [[boost_python]]]
                  ),
    [BOOST_PYTHON_LIBNAME=$withval]
  )
  AC_SUBST(BOOST_PYTHON_LIBNAME)

  ac_save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $PYTHON_INCLUDES"
  AC_CHECK_HEADER([boost/python.hpp], [
    ac_save_LIBS="$LIBS"
    LIBS="$LIBS -l$BOOST_PYTHON_LIBNAME $PYTHON_LIBS"
    AC_MSG_CHECKING([for Boost.Python runtime library availability])
    AC_TRY_LINK([
#include <boost/python/module.hpp>
    ], [
boost::python::detail::init_module("dummy", 0);
    ], [
      AC_MSG_RESULT([yes])
      AC_DEFINE(HAVE_BOOST_PYTHON, 1, [Define to 1 if Boost.Python is available])
      ifelse([$1], [],, [$1])
    ], [
      AC_MSG_RESULT([no])
      ifelse([$2], [],, [$2])
    ])
    LIBS="$ac_save_LIBS"
  ], [
    ifelse([$2], [],, [$2])
  ])
  CPPFLAGS="$ac_save_CPPFLAGS"
])

