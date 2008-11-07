ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=${includedir}/${ECELL_DIRNAME}

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@

INCLUDES = -I${top_srcdir}  -I${top_srcdir}/.. -I.. ${DMTOOL_INCLUDE} ${INCLTDL}
LIBS=../libecs.la -lboost_unit_test_framework -lgsl -lgslcblas -lstdc++
AM_CXXFLAGS=@CXXFLAGS@
AM_CPPFLAGS=-DBOOST_TEST_DYN_LINK=1 -DBOOST_TEST_MAIN=1

check_PROGRAMS=

check_PROGRAMS += DynamicPriorityQueue_test
DynamicPriorityQueue_test_SOURCES = DynamicPriorityQueue_test.cpp
DynamicPriorityQueue_test_CPPFLAGS=-DBOOST_TEST_DYN_LINK=1

check_PROGRAMS += DataPointAggregator_test
DataPointAggregator_test_SOURCES = DataPointAggregator_test.cpp

check_PROGRAMS += FullID_test
FullID_test_SOURCES = FullID_test.cpp

check_PROGRAMS += VVector_test
VVector_test_SOURCES = VVector_test.cpp
VVector_test_CPPFLAGS=-DBOOST_TEST_DYN_LINK=1

check_PROGRAMS += Util_test
Util_test_SOURCES = Util_test.cpp

check_PROGRAMS += Polymorph_test
Polymorph_test_SOURCES = Polymorph_test.cpp

check_PROGRAMS += System_test
System_test_SOURCES = System_test.cpp

TESTS=$(check_PROGRAMS)
