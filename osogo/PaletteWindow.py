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
        self.set_data('toolbar', aToolbar)

        aPluginNameList = []
        aIndicator = 0
        for aFileName in os.listdir( 'plugins' ) :

            if re.search( '\.py$', aFileName ):
                aModuleName = string.replace( aFileName, '.py', '' )

                if aIndicator == 0:
                    aIndicator = 1

                    aFirstButton = GtkRadioButton( label = aModuleName )
                    aToolbar.append_widget( aFirstButton, '', '' )
                    aPluginNameList.append( aModuleName )
                    self.set_data( aModuleName, aFirstButton )

                else :
                    aButton = GtkRadioButton( aFirstButton, label = aModuleName )
                    aToolbar.append_widget( aButton, '', '' )
                    aPluginNameList.append( aModuleName )
                    self.set_data( aModuleName, aButton )

        self.set_data( 'plugin_list' , aPluginNameList )

    def getSelectedPluginName( self ):
        aPluginList = self.get_data( 'plugin_list' )
        for aPluginName in aPluginList :
            aButton = self.get_data( aPluginName )
            if aButton.get_active() :
                aSelectedPluginName = aPluginName
        return aSelectedPluginName
        
def mainLoop():
    aPaletteWindow = PaletteWindow()
    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":

    main()
