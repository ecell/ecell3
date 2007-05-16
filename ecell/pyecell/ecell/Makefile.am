AUTOMAKE_OPTIONS = foreign

if OS_WIN32
  no_undefined = -no-undefined
  LIBPYTHON = -lpython@PYTHON_WIN_VERSION@
endif

SUBDIRS = SessionManager\
analysis

CXX_FLAGS=@CXX_FLAGS@

EXTRA_DISTS=\
__init__.py\
EntityStub.py\
LoggerStub.py\
StepperStub.py\
ObjectStub.py\
DataFileManager.py\
DataFile.py\
ECDDataFile.py\
ecs_constants.py\
ecs.py\
ecsinit.py\
emc.py\
Session.py\
ecssupport.py\
FullID.py\
RecordingSimulator.py\
Window.py\
PluginWindow.py\
Plugin.py\
util.py\
eml.py\
em.py\
TableIO.py\
emparser.py\
lex.py\
yacc.py\
emlextab.py\
emparsetab.py\
GtkSessionMonitor.py\
gui_config.py\
SbmlFunctions.py\
convertEML2SBML.py\
convertSBML2EML.py\
convertSBMLFunctions.py\
expressionlextab.py\
expressionparser.py\
expressionparsetab.py\
ModelWalker.py
