AUTOMAKE_OPTIONS = foreign

SUBDIRS = resources plugins

EXTRA_DIST = \
osogo.ini\
ecell/osogo/AboutSessionMonitor.py\
ecell/osogo/BoardWindow.py\
ecell/osogo/ConfirmWindow.py\
ecell/osogo/DataGenerator.py\
ecell/osogo/EntityListWindow.py\
ecell/osogo/FullPNQueue.py\
ecell/osogo/GtkSessionMonitor.py\
ecell/osogo/InterfaceWindow.py\
ecell/osogo/LoggerFactory.py\
ecell/osogo/LoggerWindow.py\
ecell/osogo/LoggingPolicy.py\
ecell/osogo/MainWindow.py\
ecell/osogo/MessageWindow.py\
ecell/osogo/OsogoPluginManager.py\
ecell/osogo/OsogoPluginWindow.py\
ecell/osogo/OsogoUtil.py\
ecell/osogo/OsogoWindow.py\
ecell/osogo/Plot.py\
ecell/osogo/PluginInstanceSelection.py\
ecell/osogo/PluginWindow.py\
ecell/osogo/StepperWindow.py\
ecell/osogo/VariableReferenceEditor.py\
ecell/osogo/ViewWindow.py\
ecell/osogo/Window.py\
ecell/osogo/__init__.py\
ecell/osogo/config.py\
ecell/osogo/constants.py\
ecell/osogo/main.py

EXTRA_DIST +=\
ecell3-session-monitor

all-local:
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py config
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py build

mostlyclean-local:
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py clean

install-exec-local:
	DESTDIR=$(DESTDIR); \
	sysconfdir=$(sysconfdir) \
	datarootdir=$(datarootdir) \
	libdir=$(libdir) \
	$(PYTHON) setup.py install -f --prefix=$(PREFIX) $${DESTDIR:+--root }$${DESTDIR}

