

TOPSRCDIR=${top_srcdir}
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
NUMPY_INCLUDE_DIR=@NUMPY_INCLUDE_DIR@
PYTHON_INCLUDES=@PYTHON_INCLUDES@ -I${NUMPY_INCLUDE_DIR}

ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=${includedir}/${ECELL_DIRNAME}
pkgpyexecdir=@pkgpyexecdir@

if OS_WIN32
  SUBDIRS = windows
  no_undefined = -no-undefined
  LIBPYTHON = -lpython@PYTHON_WIN_VERSION@
else
  bin_SCRIPTS =\
  ecell3-python
endif

BPLCXXFLAGS=-ftemplate-depth-70

ECELLINCLUDES=-I${TOPSRCDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHON_INCLUDES}
AM_CXXFLAGS=@CXXFLAGS@ ${ECELLINCLUDES} ${BPLCXXFLAGS}

LIBECSDIR=$(TOPSRCDIR)/libecs
LIBEMCDIR=$(TOPSRCDIR)/libemc

LIBECS=$(LIBECSDIR)/libecs.la
LIBEMC=$(LIBEMCDIR)/libemc.la
LIBBOOSTPYTHON=-lboost_python

CFLAGS=${CXXFLAGS}


pkgpyexec_LTLIBRARIES = _ecs.la _emc.la

pyecsdir=${ECELL_INCLUDE_DIR}/pyecs
pyecs_HEADERS=\
PyEcs.hpp PyEmc.hpp

_ecs_la_SOURCES=\
PyEcs.cpp

_ecs_la_LDFLAGS=-module -export-dynamic -avoid-version $(no_undefined)

_ecs_la_LIBADD=$(LIBECS) $(LIBBOOSTPYTHON) $(LIBPYTHON)

_ecs_la_DEPENDENCIES=$(LIBECS)



_emc_la_SOURCES=\
PyEmc.cpp

_emc_la_LDFLAGS=$(_ecs_la_LDFLAGS)

_emc_la_LIBADD=$(LIBECS) $(LIBEMC) $(LIBBOOSTPYTHON) $(LIBPYTHON)

_emc_la_DEPENDENCIES=$(LIBECS) $(LIBEMC)


ECS_SO=_ecs@LTDL_SHLIB_EXT@
EMC_SO=_emc@LTDL_SHLIB_EXT@

all-local: $(ECS_SO) $(EMC_SO)

$(ECS_SO): _ecs.la 
	ln -sf .libs/$(ECS_SO) ./$(ECS_SO)

$(EMC_SO): _emc.la 
	ln -sf .libs/$(EMC_SO) ./$(EMC_SO)

clean-local:
	rm -f $(ECS_SO) $(EMC_SO)


