
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

DOXYGEN_OUTFILE=doc/refman/html/index.html

doc: ${DOXYGEN_OUTFILE}

${DOXYGEN_OUTFILE}:
	doxygen ecell3.doxy

doc-clean:
	rm -rf doc/refman
