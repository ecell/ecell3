abs_top_builddir=@abs_top_builddir@
libdir=@libdir@

ECELL_DIRNAME=@ECELL_DIRNAME@
ECELL_INCLUDE_DIR=$(includedir)/$(ECELL_DIRNAME)

DMTOOL_INCLUDE = -I@DMTOOL_INCLUDE_DIR@
INCLTDL = @INCLTDL@
LIBLTDL = @LIBLTDL@

INCLUDES = -I.. -I$(top_srcdir) $(DMTOOL_INCLUDE) $(INCLTDL)
noinst_LTLIBRARIES = libecs-scripting.la

AM_CXXFLAGS = @CXXFLAGS@

libecs_scripting_la_CXXFLAGS=@CXXFLAGS@
libecs_scripting_la_SOURCES =\
ExpressionCompiler.cpp\
Assembler.cpp\
VirtualMachine.cpp

libecs_scripting_ladir=$(ECELL_INCLUDE_DIR)/libecs/scripting
libecs_scripting_la_HEADERS=\
ExpressionCompiler.hpp\
VirtualMachine.hpp\
Assembler.hpp\
Instruction.hpp
