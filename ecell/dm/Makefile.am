abs_top_builddir = @abs_top_builddir@

SUFFIXES = .cpp $(LTDL_SHLIB_EXT)

SHELL = @SHELL@

AM_CXXFLAGS = @CXXFLAGS@

ECELL3_DMC = $(top_builddir)/pyecell/ecell3-python ../bin/ecell3-dmc

ECELL_DIRNAME = @ECELL_DIRNAME@

DM_SO_DIR = ${libdir}/${ECELL_DIRNAME}/dms
DM_SRC_DIR =  ${datadir}/${ECELL_DIRNAME}/dms

PYTHON_INCLUDES = @PYTHON_INCLUDES@
PYTHON_LIBS = @PYTHON_LIBS@

DMS = \
	ConstantFluxProcess.cpp \
	DAEStepper.cpp \
	DecayFluxProcess.cpp \
	ESSYNSStepper.cpp \
	ExpressionAlgebraicProcess.cpp \
	ExpressionAssignmentProcess.cpp \
	ExpressionFluxProcess.cpp \
	FixedDAE1Stepper.cpp \
	FixedODE1Stepper.cpp \
	FluxDistributionStepper.cpp \
	GMAProcess.cpp \
	GillespieProcess.cpp \
	MassActionFluxProcess.cpp \
	MichaelisUniUniFluxProcess.cpp \
	ODE23Stepper.cpp \
	ODE45Stepper.cpp \
	ODEStepper.cpp \
	PingPongBiBiFluxProcess.cpp \
	PythonFluxProcess.cpp \
	PythonProcess.cpp \
	QuasiDynamicFluxProcess.cpp \
	SSystemProcess.cpp \
	TauLeapProcess.cpp \
	TauLeapStepper.cpp

DMH = \
	ESSYNSProcessInterface.hpp \
	PythonProcessBase.hpp \
	ExpressionProcessBase.hpp \
	GillespieProcess.hpp \
	GillespieProcessInterface.hpp \
	QuasiDynamicFluxProcess.hpp \
	QuasiDynamicFluxProcessInterface.hpp

SOS = \
	MassActionFluxProcess$(LTDL_SHLIB_EXT) \
	MichaelisUniUniFluxProcess$(LTDL_SHLIB_EXT) \
	ConstantFluxProcess$(LTDL_SHLIB_EXT) \
	DecayFluxProcess$(LTDL_SHLIB_EXT) \
	SSystemProcess$(LTDL_SHLIB_EXT) \
	GillespieProcess$(LTDL_SHLIB_EXT) \
	TauLeapProcess$(LTDL_SHLIB_EXT) \
	TauLeapStepper$(LTDL_SHLIB_EXT) \
	FixedODE1Stepper$(LTDL_SHLIB_EXT) \
	FixedDAE1Stepper$(LTDL_SHLIB_EXT) \
	ODE23Stepper$(LTDL_SHLIB_EXT) \
	ODE45Stepper$(LTDL_SHLIB_EXT) \
	DAEStepper$(LTDL_SHLIB_EXT) \
	ODEStepper$(LTDL_SHLIB_EXT) \
	ESSYNSStepper$(LTDL_SHLIB_EXT) \
	GMAProcess$(LTDL_SHLIB_EXT) \
	SSystemProcess$(LTDL_SHLIB_EXT) \
	FluxDistributionStepper$(LTDL_SHLIB_EXT) \
	QuasiDynamicFluxProcess$(LTDL_SHLIB_EXT) \
	ExpressionFluxProcess$(LTDL_SHLIB_EXT) \
	ExpressionAlgebraicProcess$(LTDL_SHLIB_EXT) \
	ExpressionAssignmentProcess$(LTDL_SHLIB_EXT)

if HAVE_PYTHON_LIB
SOS += \
	PythonFluxProcess$(LTDL_SHLIB_EXT) \
	PythonProcess$(LTDL_SHLIB_EXT)
endif

EXTRA_DIST = ${DMS} ${DMH}
dmdir = ${DM_SRC_DIR}
dm_DATA = ${DMS} ${DMH}

dmsodir = ${DM_SO_DIR}
dmso_DATA = ${DESCS}

all-local: $(SOS)

PythonFluxProcess$(LTDL_SHLIB_EXT) : PythonFluxProcess.cpp
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) $(PYTHON_LIBS) -l$(BOOST_PYTHON_LIBNAME) $(LIBS) $(EXTRA_LDFLAGS)" --cxxflags='${CPPFLAGS} ${CXXFLAGS} $(EXTRA_CXXFLAGS) -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' -o $@ $< 

PythonProcess$(LTDL_SHLIB_EXT) : PythonProcess.cpp
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) $(PYTHON_LIBS) $(EXTRA_LDFLAGS) -l$(BOOST_PYTHON_LIBNAME) $(LIBS) $(EXTRA_LDFLAGS)" --cxxflags='${CPPFLAGS} ${CXXFLAGS} $(EXTRA_CXXFLAGS) -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' -o $@ $< 

.cpp$(LTDL_SHLIB_EXT):
	${ECELL3_DMC} --verbose --dmcompile="../../dmtool" --no-stdinclude --no-stdlibdir --ldflags="-L$(abs_top_builddir)/tmp$(libdir) $(LIBS) $(EXTRA_LDFLAGS)" --cxxflags='${CPPFLAGS} ${CXXFLAGS} $(EXTRA_CXXFLAGS) -I${top_srcdir}/libecs -I${top_srcdir}/.. -I${top_srcdir} -I${top_srcdir}/../libltdl  ${PYTHON_INCLUDES} ' -o $@ $< 

# This install customization is needed because DLL's cannot be loaded by
# LoadLibrary() without executable flags set to them under Windows NT.
install-data-local:
	test -z "$(dmsodir)" || $(mkdir_p) "$(DESTDIR)$(dmsodir)"
	@for p in $(SOS); do \
		if test -f "$$p"; then d=; else d="$(srcdir)/"; fi; \
		$(am__strip_dir) \
		echo " ${INSTALL} -m 755 '$$d$$p' '$(DESTDIR)$(dmsodir)/$$f'"; \
		${INSTALL} -m 755 "$$d$$p" "$(DESTDIR)$(dmsodir)/$$f"; \
	done

uninstall-local:
	@for p in $(SOS); do \
		$(am__strip_dir) \
		echo " rm -f '$(DESTDIR)$(dmsodir)/$$f'"; \
		$(RM) "$(DESTDIR)$(dmsodir)/$$f"; \
	done

clean-local:
	${RM} -rf ${SOS}

