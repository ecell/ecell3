

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

pkgpythondir=@pkgpythondir@

BOOST_DIR=@BOOST_DIR@

BPLSRCDIR=$(BOOST_DIR)/libs/python/src
BPLINCLUDE=-I$(BOOST_DIR)

BPLCXXFLAGS=-ftemplate-depth-70

ECELLINCLUDES=-I${ECELLDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHON_INCLUDES}
CXXFLAGS=@CXXFLAGS@ ${BPLINCLUDE} ${ECELLINCLUDES} ${BPLCXXFLAGS}

LIBEMC=$(ECELLDIR)/libemc/libemc.la 
LIBECS=$(ECELLDIR)/libecs/libecs.la

CFLAGS=${CXXFLAGS}

BPLV2SRC=\
numeric.cpp\
list.cpp\
long.cpp\
dict.cpp\
tuple.cpp\
str.cpp\
aix_init_module.cpp\
from_python.cpp\
registry.cpp\
type_id.cpp\
enum.cpp\
class.cpp\
function.cpp\
inheritance.cpp\
life_support.cpp\
pickle_support.cpp\
errors.cpp\
module.cpp\
builtin_converters.cpp\
arg_to_python_base.cpp\
iterator.cpp\
object_protocol.cpp\
object_operators.cpp


BPLSRC=$(BPLV2SRC)


$(BPLSRC):
	echo $(BPLV2SRC)
	if [ -f $(BPLSRCDIR)/$@ ]; then\
		cp -f $(BPLSRCDIR)/$@ . ;\
	elif [ -f $(BPLSRCDIR)/converter/$@ ] ; then\
		cp -f $(BPLSRCDIR)/converter/$@ . ;\
	elif [ -f $(BPLSRCDIR)/object/$@ ] ; then\
		cp -f $(BPLSRCDIR)/object/$@ . ;\
	fi


#noinst_LTLIBRARIES= libbpl.la
#libbpl_la_SOURCES=$(BPLSRC)


pkgpython_LTLIBRARIES = _ecs.la

_ecs_la_SOURCES=\
PyEcs.cpp\
$(BPLSRC)


noinst_HEADERS=\
PyEcs.hpp

_ecs_la_LDFLAGS=-module -export-dynamic -avoid-version

_ecs_la_LIBADD= $(LIBEMC) $(LIBECS) -lstdc++ $(LIBLTDL)

all-local: _ecs.so


_ecs.so: _ecs.la 
	ln -sf .libs/_ecs.so ./_ecs.so

clean-local:
	rm -f _ecs.so
	rm -f $(BPLSRC)

bin_SCRIPTS =\
ecell3

