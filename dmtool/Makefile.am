
AUTOMAKE_OPTIONS=foreign

LTCONFIG = ../ltconfig
DMLTMAIN = ../ltmain.sh

SUBDIRS = include

bin_SCRIPTS = dmtool dmgen dmcompile dmlibtool

dmlibtool: $(DMLTMAIN)
	$(LTCONFIG) --disable-static --enable-win32-dll -o $@ $(DMLTMAIN)

