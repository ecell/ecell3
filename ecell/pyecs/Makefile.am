DMTOOL_INCLUDE = -I@DMTOOL_INCLUDE_DIR@
INCLTDL = @INCLTDL@
LIBLTDL = @LIBLTDL@
NUMPY_INCLUDE_DIR = @NUMPY_INCLUDE_DIR@
PYTHON_INCLUDES = @PYTHON_INCLUDES@ -I${NUMPY_INCLUDE_DIR}
PYTHON_VERSION = @PYTHON_VERSION@

ECELL_DIRNAME = @ECELL_DIRNAME@
ECELL_INCLUDE_DIR = ${includedir}/${ECELL_DIRNAME}
pkgpyexecdir = @pkgpyexecdir@

EXTRA_DIST = \
	PyEcs.hpp PyEmc.hpp\
	PyEcs.cpp PyEmc.cpp\
	ecell/__init__.py\
	ecell/ecs.py\
	ecell/emc.py\
	ecell/ecsinit.py

all-local:
	env CFLAGS="$(CPPFLAGS) $(CFLAGS) @INCLTDL@" LDFLAGS="$(LDFLAGS)" libdir=$(libdir) $(PYTHON) setup.py config
	env CFLAGS="$(CPPFLAGS) $(CFLAGS) @INCLTDL@" LDFLAGS="$(LDFLAGS)" libdir=$(libdir) $(PYTHON) setup.py build

mostlyclean-local:
	env CFLAGS="$(CPPFLAGS) $(CFLAGS) @INCLTDL@" LDFLAGS="$(LDFLAGS)" libdir=$(libdir) $(PYTHON) setup.py clean
	$(RM) -rf build

install-exec-local:
	env CFLAGS="$(CPPFLAGS) $(CFLAGS) @INCLTDL@" LDFLAGS="$(LDFLAGS)" libdir=$(libdir) $(PYTHON) setup.py install -f --prefix=$(PREFIX) --root $(DESTDIR)/
