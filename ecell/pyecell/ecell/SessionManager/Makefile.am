AUTOMAKE_OPTIONS = foreign

if OS_WIN32
  no_undefined = -no-undefined
  LIBPYTHON = -lpython@PYTHON_WIN_VERSION@
endif

CXX_FLAGS=@CXX_FLAGS@
PYTHON_INCLUDES=@PYTHON_INCLUDES@
pkgpythondir=@pkgpythondir@/SessionManager

LIBS=  # do not need gsl and gslcblas linked

pkgpython_PYTHON=\
Globus2SessionProxy.py\
Globus2SystemProxy.py\
SessionManager.py\
SessionProxy.py\
SystemProxy.py\
LocalSessionProxy.py\
LocalSystemProxy.py\
SGESessionProxy.py\
SGESystemProxy.py\
Util.py


#INCLUDES=-I. -I.. ${PYTHON_INCLUDES}

#pkgpython_LTLIBRARIES = _tableio.la

#_tableio_la_LDFLAGS=-module -export-dynamic -avoid-version $(no_undefined)
#_tableio_la_LIBADD= $(LIBPYTHON)
#_tableio_la_SOURCES = \
#_tableio.c

