

ECELLDIR=..
CXXINCLUDE=-I/usr/include/python1.6/CXX #-I@CXXINCLUDEDIR@
CXXOBJS=/usr/etc/CXX/*.o # @CXXOBJS@
PYTHONINCLUDE=/usr/include/python1.6 #-I@PYTHONINCLUDEDIR@
ECELLINCLUDES=-I${ECELLDIR} -I${ECELLDIR}/libecs -I${ECELLDIR}/emc -I${topdir}/dmtool/include
CXXFLAGS=${CXXINCLUDE} #@CXXFLAGS@
LIBLTDL=@LIBLTDL@

LIBKORANDOM=../../korandom/libkorandom.a

#noinst_LTLIBRARIES=ecs.la
#ecs_la_SOURCES=PyEcs.cpp PySimulator.cpp
#ecs_la_LDFLAGS=-module -avoid-version
#ecs_la_LIBADD=$(CXXOBJS) $(ECELLDIR)/libecs/libecs.a $(ECELLDIR)/emc/libemc.a $(LIBKORANDOM) -lltdl

noinst_PROGRAMS=ecs.so
ecs_so_SOURCES=PyEcs.cpp PySimulator.cpp
ecs_so_LDFLAGS=-shared
ecs_so_LDADD=$(CXXOBJS) $(ECELLDIR)/emc/libemc.a $(ECELLDIR)/libecs/libecs.a $(LIBKORANDOM) -lltdl



