DOC_html = users-manual
DOC_ps = users-manual.ps
DOC_pdf = users-manual.pdf
DOC_dvi = users-manual.dvi

MANUAL_TARGETS =

if BUILD_USERS_MANUAL_html
_htmldir = $(htmldir)/$(DOC_html)
_html_DATA = $(DOC_html)/*
MANUAL_TARGETS += $(DOC_html)
endif

if BUILD_USERS_MANUAL_ps
ps_DATA   = $(DOC_ps)
MANUAL_TARGETS += $(DOC_ps)
endif

if BUILD_USERS_MANUAL_pdf
pdf_DATA  = $(DOC_pdf)
MANUAL_TARGETS += $(DOC_pdf)
endif

if BUILD_USERS_MANUAL_dvi
dvi_DATA  = $(DOC_dvi)
MANUAL_TARGETS += $(DOC_dvi)
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
	stddmlib.xml \
	$(MANUAL_TARGETS)

all: $(MANUAL_TARGETS)

$(DOC_html): users-manual.xml
	$(DB2HTML) -o $@ $<

$(DOC_ps): users-manual.xml
	$(DB2PS) $<

$(DOC_pdf): users-manual.xml
	$(DB2PDF) $<

$(DOC_dvi): users-manual.xml
	$(DB2DVI) $<

distclean:
	$(RM) -f $(DOC_PS) $(DOC_PDF) $(DOC_DVI)
	$(RM) -rf $(DOC_HTML)
	$(RM) -rf *.junk *.tex
