

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

pkgpythondir=@pkgpythondir@

if OS_WIN32
  no_undefined = -no-undefined
  LIBPYTHON = -lpython22
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
LIBBOOSTPYTHON=$(BOOST_LIB) -lboost_python

CFLAGS=${CXXFLAGS}


pkgpython_LTLIBRARIES = _ecs.la

_ecs_la_SOURCES=\
PyEcs.cpp


noinst_HEADERS=\
PyEcs.hpp

_ecs_la_LDFLAGS=-module -export-dynamic -avoid-version $(no_undefined)

_ecs_la_LIBADD=$(LIBECS) $(LIBEMC) $(LIBBOOSTPYTHON) $(LIBPYTHON)

_ecs_la_DEPENDENCIES=$(LIBECS) $(LIBEMC)


all-local: _ecs@LTDL_SHLIB_EXT@


_ecs@LTDL_SHLIB_EXT@: _ecs.la 
	ln -sf .libs/_ecs@LTDL_SHLIB_EXT@ ./_ecs@LTDL_SHLIB_EXT@

clean-local:
	rm -f _ecs@LTDL_SHLIB_EXT@

bin_SCRIPTS =\
ecell3-python$(ECELL3_PYTHON_EXT)



# BPLSRC=\
# numeric.cpp\
# list.cpp\
# long.cpp\
# dict.cpp\
# tuple.cpp\
# str.cpp\
# aix_init_module.cpp\
# from_python.cpp\
# registry.cpp\
# type_id.cpp\
# enum.cpp\
# class.cpp\
# function.cpp\
# inheritance.cpp\
# life_support.cpp\
# pickle_support.cpp\
# errors.cpp\
# module.cpp\
# builtin_converters.cpp\
# arg_to_python_base.cpp\
# iterator.cpp\
# object_protocol.cpp\
# object_operators.cpp

# $(BPLSRC):
# 	echo $(BPLV2SRC)
# 	if [ -f $(BPLSRCDIR)/$@ ]; then\
# 		cp -f $(BPLSRCDIR)/$@ . ;\
# 	elif [ -f $(BPLSRCDIR)/converter/$@ ] ; then\
# 		cp -f $(BPLSRCDIR)/converter/$@ . ;\
# 	elif [ -f $(BPLSRCDIR)/object/$@ ] ; then\
# 		cp -f $(BPLSRCDIR)/object/$@ . ;\
# 	fi

