ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=${includedir}/${ECELL_DIRNAME}

DMTOOL_INCLUDE=-I@DMTOOL_INCLUDE_DIR@
INCLTDL=@INCLTDL@
LIBLTDL=@LIBLTDL@

INCLUDES = -I${top_srcdir}  -I${top_srcdir}/.. -I.. ${DMTOOL_INCLUDE} ${INCLTDL}
LIBS=../libecs.la -lboost_unit_test_framework -lgsl -lgslcblas -lstdc++
AM_CXXFLAGS=@CXXFLAGS@

noinst_PROGRAMS=

noinst_PROGRAMS+=DynamicPriorityQueue_test
DynamicPriorityQueue_test_SOURCES = DynamicPriorityQueue_test.cpp

noinst_PROGRAMS+=DataPointAggregator_test
DataPointAggregator_test_SOURCES = DataPointAggregator_test.cpp

noinst_PROGRAMS+=FullID_test
FullID_test_SOURCES = FullID_test.cpp

