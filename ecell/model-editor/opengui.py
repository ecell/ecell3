import pygtk
pygtk.require('2.0')
import gtk
import getopt
import sys

import ModelEditor

print sys.path

usageString = " ModelEditor for ecell3 modeling environment.\n (C): Keio University 2003-2004 \n Authors: Gabor Bereczki <gabor.bereczki@talk21.com>, Sylvia Kinta, Dini Karnaga\n\n Usage:\n\t-f\t:\t file to load.\n\t-h\t:\t prints this help.\n"

aFileName = None
if len( sys.argv)>1:
    try:
        optstrings = getopt.getopt( sys.argv[1:], 'f:h' )[0]
    except:
        print usageString
        sys.exit(1)
        
    if len( optstrings ) > 0:
        for anOpt in optstrings:
            if anOpt[0] == '-h':
                print usageString
                sys.exit(0)
            elif anOpt[0] == '-f':
                aFileName = anOpt[1]
        if aFileName == None:
            print usageString
            sys.exit(1)
    

g=ModelEditor.ModelEditor(aFileName)
gtk.main()

