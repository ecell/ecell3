AUTOMAKE_OPTIONS = foreign

SUBDIRS = libecs libemc pyecs pyecell bin dm frontend

ECELL_DIRNAME=@ECELL_DIRNAME@

ecellheaderdir=${includedir}/${ECELL_DIRNAME}
ecellheader_HEADERS= ecell_config.h

clean-local:
	$(RM) -rf tmp
