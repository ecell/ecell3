DMTOOL_INCLUDE = -I@DMTOOL_INCLUDE_DIR@
INCLTDL = @INCLTDL@
LIBLTDL = @LIBLTDL@
NUMPY_INCLUDE_DIR = @NUMPY_INCLUDE_DIR@
PYTHON_INCLUDES = @PYTHON_INCLUDES@ -I${NUMPY_INCLUDE_DIR}
PYTHON_VERSION = @PYTHON_VERSION@

ECELL_DIRNAME = @ECELL_DIRNAME@
ECELL_INCLUDE_DIR = ${includedir}/${ECELL_DIRNAME}
ECELL_STAGING_HOME=$(abs_top_builddir)/tmp
PYECELL_STAGING_HOME=$(ECELL_STAGING_HOME)/pyecell

PYTHON_ = env CFLAGS="$(CPPFLAGS) $(CFLAGS)" LDFLAGS="$(LDFLAGS)" INCLTDL="$(INCLTDL)" libdir=$(libdir) BOOST_PYTHON_LIBNAME=$(BOOST_PYTHON_LIBNAME) $(PYTHON)

EXTRA_DIST = \
	ecell/DataFile.py \
	ecell/DataFileManager.py \
	ecell/ECDDataFile.py \
	ecell/EntityStub.py \
	ecell/FullID.py \
	ecell/LoggerStub.py \
	ecell/ObjectStub.py \
	ecell/Plugin.py \
	ecell/RecordingSimulator.py \
	ecell/SbmlFunctions.py \
	ecell/Session.py \
	ecell/StepperStub.py \
	ecell/TableIO.py \
	ecell/_ecs.cpp\
	ecell/__init__.py \
	ecell/convertEML2SBML.py \
	ecell/convertSBML2EML.py \
	ecell/convertSBMLFunctions.py \
	ecell/ecs.py \
	ecell/emc.py \
	ecell/ecs_constants.py \
	ecell/ecssupport.py \
	ecell/em.py \
	ecell/eml.py \
	ecell/emparser.py \
	ecell/expressionlextab.py \
	ecell/expressionparser.py \
	ecell/expressionparsetab.py \
	ecell/util.py \
	ecell/_tableio.c \
	ecell/README.TableIO \
	ecell/session_manager/Constants.py \
	ecell/session_manager/Globus2Proxy.py \
	ecell/session_manager/Globus4Proxy.py \
	ecell/session_manager/LSFProxy.py \
	ecell/session_manager/LocalProxy.py \
	ecell/session_manager/SGEProxy.py \
	ecell/session_manager/SessionManager.py \
	ecell/session_manager/Util.py \
	ecell/session_manager/__init__.py \
	ecell/analysis/ControlCoefficient.py \
	ecell/analysis/Elasticity.py \
	ecell/analysis/Jacobian.py \
	ecell/analysis/MatrixIO.py \
	ecell/analysis/PathwayProxy.py \
	ecell/analysis/SBMLExporter.py \
	ecell/analysis/SBMLImporter.py \
	ecell/analysis/Structure.py \
	ecell/analysis/__init__.py \
	ecell/analysis/ecdsupport.py \
	ecell/analysis/emlsupport.py \
	ecell/analysis/emlsupport2.py \
	ecell/analysis/sbmlsupport.py \
	ecell/analysis/util.py \
	setup.py.in

bin_SCRIPTS = \
	ecell3-python \
	ecell3-session \
	ecell3-session-manager \
	ecell3-sbml2eml \
	ecell3-eml2sbml \
	ecell3-em2eml \
	ecell3-eml2em

all-local:
	$(PYTHON_) setup.py config
	$(PYTHON_) setup.py build
	$(PYTHON_) setup.py install -f --prefix=$(PYECELL_STAGING_HOME) --install-lib=$(PYECELL_STAGING_HOME)

clean-local:
	$(RM) -rf build *.exe *.zip *.dll *.pyd

mostlyclean-local:
	$(PYTHON_) setup.py clean
	$(RM) -rf $(PYECELL_STAGING_HOME)
	$(RM) -rf build

install-exec-local:
	$(PYTHON_) setup.py install -f --prefix=$(PREFIX) --install-scripts=$(bindir) $(if $(DESTDIR),--root $(DESTDIR), )
