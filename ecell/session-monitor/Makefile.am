SUBDIRS = plugins

ECELL_LIB_DIR = $(libdir)/$(ECELL_DIRNAME)

sessionmonitordir = $(ECELL_LIB_DIR)/session-monitor

sessionmonitor_DATA = \
	AboutSessionMonitor.glade \
	BoardWindow.glade \
	EntityListWindow.glade \
	FullPNQueue.glade \
	InterfaceWindow.glade \
	LoggerWindow.glade \
	LoggingPolicy.glade \
	MainWindow.glade \
	MessageWindow.glade \
	PluginInstanceSelection.glade \
	PluginWindowPopupMenu.glade \
	StepperWindow.glade \
	board.glade \
	osogo.ini \
	ecell.png \
	ecell32-1.png \
	ecell32-2.png \
	ecell32-3.png \
	ecell32-4.png \
	ecell32-5.png \
	ecell32-6.png \
	ecell32-7.png \
	ecell32-8.png \
	ecell32.png \
	ecell_large.png \
	icon_board.png \
	icon_ecell.png \
	icon_entity.png \
	icon_interface.png \
	icon_logger.png \
	icon_message.png \
	icon_start.png \
	icon_step.png \
	icon_stepper.png \
	icon_stop.png

sessionmonitor_PYTHON= \
	AboutSessionMonitor.py \
	BoardWindow.py \
	ConfirmWindow.py \
	DataGenerator.py \
	EntityListWindow.py \
	FullPNQueue.py \
	InterfaceWindow.py \
	LoggerFactory.py \
	LoggerWindow.py \
	LoggingPolicy.py \
	MainWindow.py \
	MessageWindow.py \
	OsogoPluginManager.py \
	OsogoPluginWindow.py \
	OsogoUtil.py \
	OsogoWindow.py \
	Plot.py \
	PluginInstanceSelection.py \
	StepperWindow.py \
	VariableReferenceEditor.py \
	ViewWindow.py \
	config.py \
	main.py \
	opengui.py

bin_SCRIPTS = ecell3-session-monitor gecell

EXTRA_DIST = $(sessionmonitor_DATA) $(bin_SCRIPTS)

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd

