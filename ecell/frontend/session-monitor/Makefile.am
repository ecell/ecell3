SUBDIRS = glade plugins

ECELL_STAGING_HOME=$(abs_top_builddir)/tmp
PYECELL_STAGING_HOME=$(ECELL_STAGING_HOME)/pyecell

PYTHON_ = env PYTHONPATH=$(PYECELL_STAGING_HOME) sysconfdir=$(sysconfdir) datarootdir=$(datarootdir) libdir=$(libdir) $(PYTHON)

EXTRA_DIST = \
	ecell/ui/__init__.py\
	AboutSessionMonitor.py\
	BoardWindow.py\
	ConfirmWindow.py\
	DataGenerator.py\
	EntityListWindow.py\
	FullPNQueue.py\
	GtkSessionMonitor.py\
	InterfaceWindow.py\
	LoggerFactory.py\
	LoggerWindow.py\
	LoggingPolicy.py\
	MainWindow.py\
	MessageWindow.py\
	ModelWalker.py\
	OsogoPluginManager.py\
	OsogoPluginWindow.py\
	OsogoUtil.py\
	OsogoWindow.py\
	Plot.py\
	PluginInstanceSelection.py\
	PluginWindow.py\
	StepperWindow.py\
	VariableReferenceEditor.py\
	ViewWindow.py\
	Window.py\
	__init__.py\
	config.py\
	main.py\
	opengui.py

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
