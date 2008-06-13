DOC_html = users-manual
DOC_ps = users-manual.ps
DOC_pdf = users-manual.pdf
DOC_dvi = users-manual.dvi

_htmldir = $(htmldir)/$(DOC_html)
_html_DATA = $(DOC_html)/*

psdir  = $(docdir)
pdfdir = $(docdir)
dvidir = $(docdir)

if BUILD_USERS_MANUAL_html
$(DOC_html)/*: users-manual.xml
	$(DB2HTML) -o $(DOC_html) $<
else
$(DOC_html)/*:
	$(RM) -rf $(DOC_html)
	mkdir $(DOC_html)
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
$(DOC_dvi): users-manual.xml
	$(DB2DVI) $<
endif

EXTRA_DIST = \
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

if BUILD_USERS_MANUAL_html
EXTRA_DIST += $(DOC_html)
endif

clean-local:
	$(RM) -f $(DOC_ps) $(DOC_pdf) $(DOC_dvi)
	$(RM) -rf *.junk *.tex
