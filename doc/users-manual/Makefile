all: dvi ps pdf

dvi: ecell3-users-manual.dvi

ps: ecell3-users-manual.ps

pdf: ecell3-users-manual.pdf

ecell3-users-manual.ps: ecell3-users-manual.xml
	db2ps ecell3-users-manual.xml

ecell3-users-manual.pdf: ecell3-users-manual.xml
	db2pdf ecell3-users-manual.xml

ecell3-users-manual.dvi: ecell3-users-manual.xml
	echo "Running DocBook processor..."
	db2dvi ecell3-users-manual.xml

clean:
	rm -rf *.ps *.dvi *.pdf ecell3-users-manual
