
#AUTOMAKE_OPTIONS = foreign

SUBDIRS = @ECELL_SUBDIR@

EXTRA_DIST = autogen.sh\
test


docdir = $(prefix)/share/doc/$(PACKAGE)-$(VERSION)


install-data-hook:
	$(mkinstalldirs) $(docdir)
	$(MAKE) $(AM_MAKEFLAGS) doc doc-all
	cp README NEWS COPYING INSTALL AUTHORS ChangeLog $(docdir)
	(cd doc ; cp -r * $(docdir) )
	rm -rf $(docdir)/CVS
	rm -rf $(docdir)/*/CVS
	rm -rf $(docdir)/*/*/CVS

uninstall-hook:
	rm -rf $(docdir)


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

doc-all: doc-ps doc-pdf doc-html-archive

doc-ps: doc
	make -C ${DOXYGEN_DIR}/latex ps
	cp ${DOXYGEN_DIR}/latex/refman.ps ${DOXYGEN_DIR}
	make -C ${DOCBOOK_DIR} ps

doc-pdf: doc
	make -C ${DOXYGEN_DIR}/latex pdf
	cp ${DOXYGEN_DIR}/latex/refman.pdf ${DOXYGEN_DIR}
	make -C ${DOCBOOK_DIR} pdf

doc-html-archive: doc
	(cd ${DOXYGEN_DIR} ; tar -czf refman.html.tgz html/*)

doc-clean:
	rm -rf doc/refman/html/*
	rm -rf doc/refman/latex/*
	rm -rf ${DOCBOOK_DIR}/ecell3-users-manual
	rm -f *.dvi *.ps *.pdf


${DOXYGEN_OUTFILE}:
	doxygen ecell3.doxy

${DOCBOOK_OUTFILE}:
	db2html ${DOCBOOK_DIR}/ecell3-users-manual.xml

