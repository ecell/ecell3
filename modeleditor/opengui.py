import ModelEditor
import gtk
import getopt
import sys
usageString = " ModelEditor for ecell3 modeling environement.\n (C): Keio University 2003 \n Author: Gabor Bereczki <gabor.bereczki@talk21.com>\n\n Usage:\n\t-f\t:\t file to load.\n\t-h\t:\t prints this help.\n"

aFileName = None
optstrings = getopt.getopt( sys.argv[1:], 'f:h' )[0]

for anOpt in optstrings:
	if anOpt[0] == '-h':
		print usageString
		sys.exit(0)
	elif anOpt[0] == '-f':
		aFileName = anOpt[1]
	else:
		print usageString
		sys.exit(1)

g=ModelEditor.ModelEditor(aFileName)
gtk.mainloop()

