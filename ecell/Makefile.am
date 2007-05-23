
AUTOMAKE_OPTIONS = foreign

SUBDIRS = libecs libemc pyecs pyecell bin dm
if WITH_GTK
SUBDIRS += session-monitor model-editor toollauncher
endif

ECELL_DIRNAME=@ECELL_DIRNAME@

ecellheaderdir=${includedir}/${ECELL_DIRNAME}
ecellheader_HEADERS= ecell_config.h
