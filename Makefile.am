
#AUTOMAKE_OPTIONS = foreign

SUBDIRS = @ECELL_SUBDIR@

EXTRA_DIST = autogen.sh\
test


docdir = $(prefix)/share/doc/$(PACKAGE)-$(VERSION)


install-data-hook:
	$(mkinstalldirs) $(docdir)
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

doc: ${DOXYGEN_OUTFILE}

doc-all: doc-ps doc-pdf doc-html-archive

doc-ps: doc
	make -C ${DOXYGEN_DIR}/latex ps
	cp ${DOXYGEN_DIR}/latex/refman.ps ${DOXYGEN_DIR}

doc-pdf: doc
	make -C ${DOXYGEN_DIR}/latex pdf
	cp ${DOXYGEN_DIR}/latex/refman.pdf ${DOXYGEN_DIR}

doc-html-archive: doc
	(cd ${DOXYGEN_DIR} ; tar -czf refman.html.tgz html/*)

doc-clean:
	rm -rf doc/refman/html/*
	rm -rf doc/refman/latex/*


${DOXYGEN_OUTFILE}:
	doxygen ecell3.doxy


