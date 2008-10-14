AUTOMAKE_OPTIONS = foreign

SUBDIRS = libecs pyecell bin dm frontend

ECELL_DIRNAME = @ECELL_DIRNAME@

ecellheaderdir = ${includedir}/${ECELL_DIRNAME}
ecellheader_HEADERS = ecell_config.h

EXTRA_DIST = configure.ac.in

clean-local:
	$(RM) -rf tmp
