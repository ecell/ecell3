

ECELLDIR=..
DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@
PYTHON_INCLUDES=@PYTHON_INCLUDES@

ecellpyexecdir=$(pyexecdir)/ecell

BOOST_DIR=@BOOST_DIR@

BPLSRCDIR=$(BOOST_DIR)/libs/python/src
BPLINCLUDE=-I$(BOOST_DIR)

BPLCXXFLAGS=-ftemplate-depth-21

ECELLINCLUDES=-I${ECELLDIR} ${DMTOOL_INCLUDE} ${INCLTDL} ${PYTHON_INCLUDES}
CXXFLAGS=@CXXFLAGS@ ${BPLINCLUDE} ${ECELLINCLUDES} ${BPLCXXFLAGS}

LIBEMC=$(ECELLDIR)/libemc/libemc.la 
LIBECS=$(ECELLDIR)/libecs/libecs.la

CFLAGS=${CXXFLAGS}

#FIXME: _ecs.la cannot be used with automake-1.4.

BPLV1SRC=\
classes.cpp\
conversions.cpp\
cross_module.cpp\
errors.cpp\
extension_class.cpp\
functions.cpp\
init_function.cpp\
module_builder.cpp\
objects.cpp\
types.cpp

BPLV2SRC=\
builtin_converters.cpp\
callback.cpp\
from_python.cpp\
registry.cpp\
type_id.cpp\
class.cpp\
function.cpp\
inheritance.cpp\
life_support.cpp

BPLSRC=$(BPLV1SRC)


$(BPLSRC):
	cp -f $(BPLSRCDIR)/$@ .
#	cp -f $(BPLSRCDIR)/*.cpp .
#	cp -f $(BPLSRCDIR)/converter/*.cpp .
#	cp -f $(BPLSRCDIR)/object/*.cpp .
#	cp -f $(BPLSRCDIR)/detail/*.cpp .


ecellpyexec_LTLIBRARIES = ecs.la

ecs_la_SOURCES=\
PyEcs.cpp\
$(BPLSRC) 


pyecs_HEADERS=\
PyEcs.hpp

ecs_la_LDFLAGS=-module -export-dynamic -avoid-version

ecs_la_LIBADD= $(LIBEMC) $(LIBECS) -lstdc++ $(LIBLTDL)

all-local: ecs.so

#FIXME: should be removed after ecs.la is renamed to _ecs.la
#FIXME: messy
install-exec-local:
	mv -f $(ecellpyexecdir)/ecs.so $(ecellpyexecdir)/_ecs.so
	mv -f $(ecellpyexecdir)/ecs.la $(ecellpyexecdir)/_ecs.la


ecs.so: ecs.la 
	ln -sf .libs/ecs.so ./ecs.so

clean-local:
	rm -f ecs.so _ecs.so
	rm -f $(BPLSRC)

pyecsdir=${pkgincludedir}/pyecs

bin_SCRIPTS =\
ecell3

