SUBDIRS = users-manual samples

EXTRA_DIST = \
	ecell3.doxy.in \
	index.tmpl

if BUILD_API_REF_html
_htmldir = $(htmldir)/api
_html_DATA = api/*

api/*: ecell3.doxy
	doxygen ecell3.doxy

clean-local:
	$(RM) -r api

distclean-local:
	$(RM) -f api
endif
