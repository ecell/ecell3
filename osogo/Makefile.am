AUTOMAKE_OPTIONS = foreign

SUBDIRS = plugins

osogodir=${pkglibdir}
osogo_PYTHON=\
osogo.ini\
BoardWindow.py\
BoardWindow.glade\
ConfirmWindow.py\
DataGenerator.py\
EntityListWindow.glade\
EntityListWindow.py\
FullPNQueue.py\
FullPNQueue.glade\
InterfaceWindow.py\
InterfaceWindow.glade\
LoggerWindow.glade\
LoggerWindow.py\
LoggingPolicy.py\
LoggingPolicy.glade\
MainWindow.glade\
MainWindow.py\
MessageWindow.glade\
MessageWindow.py\
OsogoWindow.py\
OsogoPluginManager.py\
OsogoPluginWindow.py\
StepperWindow.glade\
StepperWindow.py\
AboutSessionMonitor.glade\
AboutSessionMonitor.py\
OsogoUtil.py\
PluginInstanceSelection.glade\
PluginInstanceSelection.py\
PluginWindowPopupMenu.glade\
VariableReferenceEditor.py\
Plot.py\
board.glade\
config.py\
main.py\
opengui.py\
ecell.png\
ecell32.png\
ecell_large.png\
icon_board.png\
icon_ecell.png\
icon_entity.png\
icon_interface.png\
icon_logger.png\
icon_message.png\
icon_start.png\
icon_step.png\
icon_stepper.png\
icon_stop.png


EXTRA_DIST=${osogo_PYTHON} ${BINS}

# Windows py2exe setup options, ignore for Unix based systems
PY2EXE_SETUP = ../ecell/pyecs/windows/py2exeSetup
WIN_ICON = plugins/ecell.ico
PATH_VARIABLES = ../ecell/pyecs/windows/pathVariables

BINS =\
gecell

EXE = ${BINS:=$(EXEEXT)}
bin_SCRIPTS = ${EXE}

%.exe : %
	python $(PY2EXE_SETUP) --source=$< --icon=$(WIN_ICON) --merge=$(PATH_VARIABLES) --gui py2exe

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd

