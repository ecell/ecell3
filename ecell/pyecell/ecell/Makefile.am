AUTOMAKE_OPTIONS = foreign

CXX_FLAGS=@CXX_FLAGS@
PYTHON_INCLUDES=@PYTHON_INCLUDES@
pkgpythondir=@pkgpythondir@

LIBS=  # do not need gsl and gslcblas linked

pkgpython_PYTHON=\
__init__.py\
EntityStub.py\
LoggerStub.py\
StepperStub.py\
ObjectStub.py\
DataFileManager.py\
DataFile.py\
ECDDataFile.py\
ECS.py\
ecs.py\
Session.py\
ecssupport.py\
FullID.py\
Plugin.py\
PluginWindow.py\
RecordingSimulator.py\
util.py\
Window.py\
eml.py\
em.py\
spark.py\
TableIO.py

INCLUDES=-I. -I.. ${PYTHON_INCLUDES}

pkgpython_LTLIBRARIES = _tableio.la

_tableio_la_LDFLAGS=-module -export-dynamic -avoid-version
_tableio_la_LIBADD=
_tableio_la_SOURCES = \
_tableio.c

