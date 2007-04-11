
#AUTOMAKE_OPTIONS = foreign

SUBDIRS = libltdl dmtool ecell

EXTRA_DIST = autogen.sh\
depcomp\
test


docdir = $(prefix)/share/doc/$(PACKAGE)-$(VERSION)


install-data-hook:
	$(mkinstalldirs) $(DESTDIR)$(docdir)
	cp README NEWS COPYING INSTALL AUTHORS ChangeLog $(DESTDIR)$(docdir)
	rm -rf $(DESTDIR)$(docdir)/CVS
	rm -rf $(DESTDIR)$(docdir)/*/CVS
	rm -rf $(DESTDIR)$(docdir)/*/*/CVS

uninstall-hook:
	rm -rf $(DESTDIR)$(docdir)


dist-hook:
	cp -r doc $(distdir)
	rm -rf $(distdir)/CVS
	rm -rf $(distdir)/*/CVS
	rm -rf $(distdir)/*/*/CVS


DOXYGEN_DIR=doc/refman
DOXYGEN_OUTFILE=${DOXYGEN_DIR}/html/index.html
DOCBOOK_DIR=doc/users-manual
DOCBOOK_OUTFILE=${DOCBOOK_DIR}/ecell3-users-manual.html

doc: ${DOXYGEN_OUTFILE} ${DOCBOOK_OUTFILE}

doc-install:
	$(MAKE) $(AM_MAKEFLAGS) doc doc-all
	(cd doc ; cp -r * $(DESTDIR)$(docdir) )
	rm -rf $(DESTDIR)$(docdir)/CVS
	rm -rf $(DESTDIR)$(docdir)/*/CVS
	rm -rf $(DESTDIR)$(docdir)/*/*/CVS

doc-all: doc-pdf doc-html-archive

doc-pdf: doc
	make -C ${DOXYGEN_DIR}/latex
	cp ${DOXYGEN_DIR}/latex/refman.pdf ${DOXYGEN_DIR}
	make -C ${DOCBOOK_DIR} pdf

doc-html-archive: doc
	(cd ${DOXYGEN_DIR} ; tar -czf refman.html.tgz html/*)

doc-clean:
	rm -rf doc/refman/html/*
	rm -rf doc/refman/latex/*
	rm -rf ${DOCBOOK_DIR}/ecell3-users-manual
	rm -f *.dvi *.ps *.pdf

strip:
	strip ecell/dm/*.so
	strip ecell/libecs/.libs/libecs.so
	strip ecell/libemc/.libs/libemc.so
	strip ecell/pyecs/.libs/_ecs.so


${DOXYGEN_OUTFILE}:
	doxygen ecell3.doxy

${DOCBOOK_OUTFILE}:
	db2html ${DOCBOOK_DIR}/ecell3-users-manual.xml

