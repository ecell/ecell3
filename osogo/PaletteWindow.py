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
                aButtonName = string.replace( aModuleName, 'Window', '' )
                aPluginNameList.append( aModuleName )

                aPixMap = GtkPixmap( self, "plugins/" + aModuleName + '.xpm' )

                if aIndicator == 0:
                    aIndicator = 1
                    aFirstButtonObj = GtkRadioButton()
                    aFirstButton = \
                            aToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
                                                 aFirstButtonObj, aButtonName,
                                                 '', '', aPixMap, None )
                    self.set_data( aModuleName, aFirstButton )
                else :
                    aButtonObj = GtkRadioButton( aFirstButtonObj )
                    aButton = \
                            aToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
                                                     aButtonObj, aButtonName,
                                                     '', '', aPixMap, None )
                    self.set_data( aModuleName, aButton )
        self.set_data( 'plugin_list' , aPluginNameList )
        aFirstButton.set_active( 1 )

    def getSelectedPluginName( self ):
        aPluginList = self.get_data( 'plugin_list' )
        for aPluginName in aPluginList :
            aButton = self.get_data( aPluginName )
            print aPluginName
            print aButton.get_active()
            print
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
