
AUTOMAKE_OPTIONS=foreign

LTCONFIG = ../ltconfig
DMLTMAIN = ../ltmain.sh

bin_SCRIPTS = dmtool dmgen dmcompile dmlibtool

dmlibtool: $(DMLTMAIN)
	$(LTCONFIG) --disable-static --enable-win32-dll -o $@ $(DMLTMAIN)


pkginclude_HEADERS=DynamicModule.hpp ModuleMaker.hpp

#EXTRA_DIST = $(bin_SCRIPTS)
