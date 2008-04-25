AUTOMAKE_OPTIONS = foreign

SUBDIRS = libecs pyecell bin dm
if ENABLE_GUI
SUBDIRS += session-monitor
endif

ECELL_DIRNAME=@ECELL_DIRNAME@

ecellheaderdir=${includedir}/${ECELL_DIRNAME}
ecellheader_HEADERS= ecell_config.h

clean-local:
	$(RM) -rf tmp
