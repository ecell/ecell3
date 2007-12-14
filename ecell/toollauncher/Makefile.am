SUBDIRS = misc

ECELL_LIB_DIR = $(libdir)/$(ECELL_DIRNAME)

toollauncherdir = $(ECELL_LIB_DIR)/toollauncher
toollauncher_PYTHON = \
	ToolLauncher.glade \
	MessageWindow.glade \
	ToolLauncherErrorMessage.glade \
	ToolLauncherOkCancel.glade \
	ToolLauncherPreferences.glade \
	AboutToolLauncher.glade \
	config.py \
	ParentWindow.py \
	ToolLauncher.py \
	ToolLauncherErrorMessage.py \
	ToolLauncherOkCancel.py \
	ToolLauncherPreferences.py \
	AboutToolLauncher.py \
	Window.py \
	toollauncher.png \
	toollauncher32.png \
	toollauncher_large.png

bin_SCRIPTS = ecell3-toollauncher

EXTRA_DIST = $(toollauncher_PYTHON) $(bin_SCRIPTS)

clean-local:
	${RM} -rf build *.exe *.zip *.dll *.pyd

