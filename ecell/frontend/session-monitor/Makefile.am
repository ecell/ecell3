SUBDIRS = glade plugins

PYTHON_ = env PYTHONPATH=$(top_srcdir)/pyecell sysconfdir=$(sysconfdir) datarootdir=$(datarootdir) libdir=$(libdir) $(PYTHON)

EXTRA_DIST = \
	ecell/ui/__init__.py\
	ecell/ui/osogo/AboutSessionMonitor.py\
	ecell/ui/osogo/BoardWindow.py\
	ecell/ui/osogo/ConfirmWindow.py\
	ecell/ui/osogo/DataGenerator.py\
	ecell/ui/osogo/Dialog.py\
	ecell/ui/osogo/EntityListWindow.py\
	ecell/ui/osogo/FullPNQueue.py\
	ecell/ui/osogo/GtkSessionMonitor.py\
	ecell/ui/osogo/InterfaceWindow.py\
	ecell/ui/osogo/LoggerFactory.py\
	ecell/ui/osogo/LoggerWindow.py\
	ecell/ui/osogo/LoggingPolicy.py\
	ecell/ui/osogo/MainWindow.py\
	ecell/ui/osogo/MessageWindow.py\
	ecell/ui/osogo/ModelWalker.py\
	ecell/ui/osogo/OsogoPluginManager.py\
	ecell/ui/osogo/OsogoPluginWindow.py\
	ecell/ui/osogo/OsogoWindow.py\
	ecell/ui/osogo/Pane.py\
	ecell/ui/osogo/Plot.py\
	ecell/ui/osogo/PluginInstanceSelection.py\
	ecell/ui/osogo/PluginWindow.py\
	ecell/ui/osogo/SeparativePane.py\
	ecell/ui/osogo/StepperWindow.py\
	ecell/ui/osogo/VariableReferenceEditor.py\
	ecell/ui/osogo/ViewWindow.py\
	ecell/ui/osogo/Window.py\
	ecell/ui/osogo/__init__.py\
	ecell/ui/osogo/config.py\
	ecell/ui/osogo/constants.py\
	ecell/ui/osogo/main.py\
	ecell/ui/osogo/utils.py

EXTRA_DIST +=\
	osogo.ini\
	setup.py

EXTRA_DIST +=\
	ecell3-session-monitor

all-local:
	$(PYTHON_) setup.py config
	$(PYTHON_) setup.py build

mostlyclean-local:
	$(PYTHON_) setup.py clean

install-exec-local:
	$(PYTHON_) setup.py install -f --prefix=$(PREFIX) $(if $(DESTDIR),--root $(DESTDIR), )
