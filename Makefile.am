
#AUTOMAKE_OPTIONS = foreign

SUBDIRS = libltdl korandom dmtool ecell osogo

EXTRA_DIST = autogen.sh\
PyCXX\
doc\
test

dist-hook:
	rm -rf $(distdir)/CVS
	rm -rf $(distdir)/*/CVS
	rm -rf $(distdir)/*/*/CVS

DOXYGEN_OUTFILE=doc/refman/libecs/html/index.html

doc: ${DOXYGEN_OUTFILE}

${DOXYGEN_OUTFILE}:
	doxygen ecell3_libecs.doxy
	doxygen ecell3_libemc.doxy

doc-clean:
	rm -rf doc/refman/libecs
	rm -rf doc/refman/libemc
