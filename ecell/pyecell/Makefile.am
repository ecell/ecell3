AUTOMAKE_OPTIONS = foreign # 1.5

all-local:
	$(PYTHON) setup.py config
	$(PYTHON) setup.py build

mostlyclean-local:
	$(PYTHON) setup.py clean

install-exec-local:
	DESTDIR=$(DESTDIR); \
	$(PYTHON) setup.py install -f --prefix=$(PREFIX) $${DESTDIR:+--root }$${DESTDIR}

EXTRA_DIST =\
ecell3-session\
ecell3-session-manager\
ecell3-session-gui\
ecell3-sbml2eml\
ecell3-eml2sbml\
ecell3-em2eml\
ecell3-eml2em

EXTRA_DIST +=\
ecell/DataFile.py\
ecell/DataFileManager.py\
ecell/ECDDataFile.py\
ecell/EntityStub.py\
ecell/FullID.py\
ecell/GtkSessionMonitor.py\ ecell/LoggerStub.py\
ecell/ModelWalker.py\
ecell/ObjectStub.py\
ecell/Plugin.py\
ecell/PluginWindow.py\
ecell/RecordingSimulator.py\
ecell/SbmlFunctions.py\
ecell/Session.py\
ecell/StepperStub.py\
ecell/TableIO.py\
ecell/Window.py\
ecell/__init__.py\
ecell/convertEML2SBML.py\
ecell/convertSBML2EML.py\
ecell/convertSBMLFunctions.py\
ecell/ecs_constants.py\
ecell/ecssupport.py\
ecell/em.py\
ecell/eml.py\
ecell/emlextab.py\
ecell/emparser.py\
ecell/emparsetab.py\
ecell/expressionlextab.py\
ecell/expressionparser.py\
ecell/expressionparsetab.py\
ecell/gui_config.py\
ecell/util.py\
ecell/_tableio.c\
ecell/README.TableIO\
ecell/SessionManager/Globus2SessionProxy.py\
ecell/SessionManager/Globus2SystemProxy.py\
ecell/SessionManager/LocalSessionProxy.py\
ecell/SessionManager/LocalSystemProxy.py\
ecell/SessionManager/SGESessionProxy.py\
ecell/SessionManager/SGESystemProxy.py\
ecell/SessionManager/SessionManager.py\
ecell/SessionManager/SessionProxy.py\
ecell/SessionManager/SystemProxy.py\
ecell/SessionManager/Util.py\
ecell/SessionManager/__init__.py\
ecell/analysis/ControlCoefficient.py\
ecell/analysis/Elasticity.py\
ecell/analysis/Jacobian.py\
ecell/analysis/MatrixIO.py\
ecell/analysis/PathwayProxy.py\
ecell/analysis/SBMLExporter.py\
ecell/analysis/SBMLImporter.py\
ecell/analysis/Structure.py\
ecell/analysis/__init__.py\
ecell/analysis/ecdsupport.py\
ecell/analysis/emlsupport.py\
ecell/analysis/emlsupport2.py\
ecell/analysis/sbmlsupport.py\
ecell/analysis/util.py\
setup.py.in
