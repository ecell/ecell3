abs_top_builddir=@abs_top_builddir@
libdir=@libdir@

ECELLDIR=..

ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=${includedir}/${ECELL_DIRNAME}

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@

LIBECS=$(ECELLDIR)/libecs/libecs.la

INCLUDES = -I${top_srcdir} ${DMTOOL_INCLUDE} ${INCLTDL}
lib_LTLIBRARIES = libemc.la

AM_CXXFLAGS = $(CXXFLAGS)

libemc_la_LDFLAGS= -version-info 2:0:0 -no-undefined

libemc_la_LIBADD=$(LIBECS) -lstdc++ 
libemc_la_DEPENDENCIES=$(LIBECS)
libemc_la_SOURCES=\
libemc.cpp\
Simulator.cpp\
SimulatorImplementation.cpp\
LocalSimulatorImplementation.cpp

libemcdir=${ECELL_INCLUDE_DIR}/libemc
libemc_HEADERS=\
libemc.hpp\
LocalSimulatorImplementation.hpp\
Simulator.hpp\
SimulatorImplementation.hpp\
SimulatorMaker.hpp

ECELL_STAGING_HOME=$(abs_top_builddir)/tmp
OUTPUT=$(ECELL_STAGING_HOME)$(libdir)/libemc$(SHLEXT)
all-local: $(OUTPUT)

$(OUTPUT): libemc.la
	test -d $(ECELL_STAGING_HOME)$(libdir) || mkdir -p -- $(ECELL_STAGING_HOME)$(libdir)
	$(LIBTOOL) --mode=install cp $< $(ECELL_STAGING_HOME)$(libdir)
