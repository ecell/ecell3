#!/usr/bin/env python

from main import *
from gtk import *
import os
import re
import string

class PaletteWindow(GtkWindow):
    def __init__( self ):
        GtkWindow.__init__(self, WINDOW_TOPLEVEL)

        self.theToolbar = GtkToolbar(ORIENTATION_VERTICAL, TOOLBAR_BOTH )
        self.add(self.theToolbar)

        aIndicator = 0
        for aFileName in os.listdir('plugins') :

            if re.search('\.py$', aFileName):
                aModuleName = string.replace(aFileName, '.py', '')

                if aIndicator == 0:
                    aIndicator = 1

                    aFirstButton = GtkRadioButton(label = aModuleName)
                    self.theToolbar.append_widget( aFirstButton, '', '' )

                else :
                    aButton = GtkRadioButton(aFirstButton, label = aModuleName)
                    self.theToolbar.append_widget( aButton, '', '' )
            
        self.show_all()

def mainLoop():
    aPaletteWindow = PaletteWindow()
    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":

    main()
