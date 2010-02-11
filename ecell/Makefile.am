AUTOMAKE_OPTIONS = foreign

SUBDIRS = libecs libemc bin dm
if ENABLE_PYECELL
SUBDIRS += pyecell frontend
endif

ECELL_DIRNAME = @ECELL_DIRNAME@

ecellheaderdir = ${includedir}/${ECELL_DIRNAME}
ecellheader_HEADERS = ecell_config.h

EXTRA_DIST = configure.ac.in

clean-local:
	$(RM) -rf tmp
