SUBDIRS = users-manual samples

EXTRA_DIST = \
	ecell3.doxy.in \
	index.tmpl

if BUILD_API_REF_html
all-local: api/

apidoc_dir=$(htmldir)/api

api/: ecell3.doxy
	doxygen ecell3.doxy

install-data-local:
	@$(NORMAL_INSTALL)
	$(MKDIR_P) $(DESTDIR)$(apidoc_dir)
	$(INSTALL_DATA) api/* $(DESTDIR)$(apidoc_dir)

uninstall-local:
	@$(NORMAL_UNINSTALL)
	rm -f $(DESTDIR)$(apidoc_dir)/*

clean-local:
	$(RM) -r api

distclean-local:
	$(RM) -f api
endif
