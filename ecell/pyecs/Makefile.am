DMTOOL_INCLUDE = -I@DMTOOL_INCLUDE_DIR@
INCLTDL = @INCLTDL@
LIBLTDL = @LIBLTDL@
NUMPY_INCLUDE_DIR = @NUMPY_INCLUDE_DIR@
PYTHON_INCLUDES = @PYTHON_INCLUDES@ -I${NUMPY_INCLUDE_DIR}
PYTHON_VERSION = @PYTHON_VERSION@

ECELL_DIRNAME = @ECELL_DIRNAME@
ECELL_INCLUDE_DIR = ${includedir}/${ECELL_DIRNAME}
pkgpyexecdir = @pkgpyexecdir@

bin_SCRIPTS = ecell3-python

EXTRA_DIST = \
	PyEcs.hpp PyEmc.hpp\
	PyEcs.cpp PyEmc.cpp\
	ecell/__init__.py\
	ecell/ecs.py\
	ecell/emc.py\
	ecell/ecsinit.py

all-local:
	CFLAGS="$(CPPFLAGS) $(CFLAGS)" \
	LDFLAGS="$(LDFLAGS)" \
	libdir="$(libdir)" \
	$(PYTHON) setup.py config
	CFLAGS="$(CPPFLAGS) $(CFLAGS)" \
	LDFLAGS="$(LDFLAGS)" \
	libdir="$(libdir)" \
	$(PYTHON) setup.py build

mostlyclean-local:
	CFLAGS="$(CPPFLAGS) $(CFLAGS)" \
	LDFLAGS="$(LDFLAGS)" \
	libdir="$(libdir)" \
	$(PYTHON) setup.py clean
	rm -rf build

install-exec-local:
	DESTDIR=$(DESTDIR); \
	CFLAGS="$(CPPFLAGS) $(CFLAGS)" \
	LDFLAGS="$(LDFLAGS)" \
	libdir="$(libdir)" \
	$(PYTHON) setup.py install -f --prefix=$(PREFIX) $${DESTDIR:+--root }$${DESTDIR}
