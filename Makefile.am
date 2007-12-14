SUBDIRS = libltdl dmtool ecell osdep doc build

BASIC_DOCS =
	AUTHORS \
	COPYING \
	INSTALL \
	NEWS \
	README \
	ChangeLog

EXTRA_DIST = \
	$(BASIC_DOCS) \
	autogen.sh \
	autodoc.sh \
	ecell_version.sh

docdir = $(prefix)/share/doc/$(PACKAGE)-$(VERSION)


install-data-hook:
	$(mkinstalldirs) $(DESTDIR)$(docdir)
	for file in $(BASIC_DOCS); do \
		$(INSTALL_DATA) $$file $(DESTDIR)$(docdir); \
	done

uninstall-hook:
	$(RM) -rf $(DESTDIR)$(docdir)
