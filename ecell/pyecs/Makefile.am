

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

pkgpythondir=@pkgpythondir@

if OS_WIN32
  no_undefined = -no-undefined
  LIBPYTHON = -lpython@PYTHON_WIN_VERSION@
  ECELL3_PYTHON_EXT = .bat
else
  ECELL3_PYTHON_EXT =
endif

BOOST_INCLUDE=@BOOST_INCLUDE@
BOOST_LIB=@BOOST_LIB@

BPLCXXFLAGS=-ftemplate-depth-70

ECELLINCLUDES=-I${ECELLDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHON_INCLUDES}
CXXFLAGS=@CXXFLAGS@ ${BOOST_INCLUDE} ${ECELLINCLUDES} ${BPLCXXFLAGS}

LIBECSDIR=$(ECELLDIR)/libecs
LIBEMCDIR=$(ECELLDIR)/libemc

LIBECS=$(LIBECSDIR)/libecs.la
LIBEMC=$(LIBEMCDIR)/libemc.la
LIBBOOSTPYTHON=$(BOOST_LIB) @BOOST_PYTHON_LIB@

CFLAGS=${CXXFLAGS}


pkgpython_LTLIBRARIES = _ecs.la _emc.la

_ecs_la_SOURCES=\
PyEcs.cpp


pyecsdir=${pkgincludedir}/pyecs
pyecs_HEADERS=\
PyEcs.hpp PyEmc.hpp

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


bin_SCRIPTS =\
ecell3-python$(ECELL3_PYTHON_EXT)

