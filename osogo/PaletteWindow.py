#!/usr/bin/env python

from main import *
from gtk import *
import os
import re
import string

class PaletteWindow(GtkWindow):
    def __init__( self ):
        GtkWindow.__init__( self, WINDOW_TOPLEVEL )

        aToolbar = GtkToolbar( ORIENTATION_VERTICAL, TOOLBAR_BOTH )
        self.add( aToolbar )

        aIndicator = 0
        for aFileName in os.listdir( 'plugins' ) :

            if re.search( '\.py$', aFileName ):
                aModuleName = string.replace( aFileName, '.py', '' )

                if aIndicator == 0:
                    aIndicator = 1

                    aFirstButton = GtkRadioButton( label = aModuleName )
                    aToolbar.append_widget( aFirstButton, '', '' )

                else :
                    aButton = GtkRadioButton( aFirstButton, label = aModuleName )
                    aToolbar.append_widget( aButton, '', '' )
            
def mainLoop():
    aPaletteWindow = PaletteWindow()
    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":

    main()
