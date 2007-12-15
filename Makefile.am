SUBDIRS = libltdl dmtool ecell osdep doc build

BASIC_DOCS = \
	AUTHORS \
	COPYING \
	INSTALL \
	NEWS \
	README \
	ChangeLog

EXTRA_DIST = \
	$(BASIC_DOCS) \
	autogen.sh \
	ecell_version.sh

doc_DATA = $(BASIC_DOCS)

dist-hook:
	cp py-compile $(distdir)
