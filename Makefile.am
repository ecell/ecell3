
AUTOMAKE_OPTIONS = foreign

SUBDIRS = libltdl korandom dmtool ecell osogo

EXTRA_DIST = autogen.sh\
PyCXX

dist-hook:
	rm -rf $(distdir)/CVS
	rm -rf $(distdir)/*/CVS
	rm -rf $(distdir)/*/*/CVS

doc:
	doxygen ecell3.doxy
