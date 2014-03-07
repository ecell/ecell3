DOC_html = users-manual
DOC_ps = users-manual.ps
DOC_pdf = users-manual.pdf
DOC_dvi = users-manual.dvi

if BUILD_USERS_MANUAL_html
all-local: $(DOC_html)/index.html

$(DOC_html)/index.html: users-manual.xml
	$(DB2HTML) -o $(DOC_html) $<

install-data-local:
	@$(NORMAL_INSTALL)
	$(MKDIR_P) $(DESTDIR)$(htmldir)/$(DOC_html)
	$(INSTALL_DATA) $(DOC_html)/* $(DESTDIR)$(htmldir)/$(DOC_html)

uninstall-local:
	@$(NORMAL_UNINSTALL)
	rm -f $(DESTDIR)$(htmldir)/$(DOC_html)/*
endif

if BUILD_USERS_MANUAL_ps
ps_DATA   = $(DOC_ps)
$(DOC_ps): users-manual.xml
	$(DB2PS) $<
endif

if BUILD_USERS_MANUAL_pdf
pdf_DATA  = $(DOC_pdf)
$(DOC_pdf): users-manual.xml
	$(DB2PDF) $<
endif

if BUILD_USERS_MANUAL_dvi
dvi_DATA  = $(DOC_dvi)
if BUILD_USERS_MANUAL_pdf
$(DOC_dvi): users-manual.xml $(DOC_pdf)
	$(DB2DVI) $<
else
$(DOC_dvi): users-manual.xml
	$(DB2DVI) $<
endif
endif

EXTRA_DIST = \
	users-manual \
	cpp-basics.xml \
	dm.xml \
	ecddatafile-api.xml \
	users-manual.xml \
	empy-manual.xml \
	inside-ecell.xml \
	intro.xml \
	legal.xml \
	modeling.xml \
	modelingtutorial.xml \
	objectstubs-api.xml \
	python-basics.xml \
	running.xml \
	scripting.xml \
	session-class-api.xml \
	stddmlib.xml

clean-local:
	$(RM) -f $(DOC_ps) $(DOC_pdf) $(DOC_dvi)
	$(RM) -rf *.junk *.tex $(DOC_html)
