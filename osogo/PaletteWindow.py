#!/usr/bin/env python

from config import *

from main import *
from gtk import *
import os
import re
import string


class PaletteWindow(GtkWindow):
    def __init__( self ):
        GtkWindow.__init__( self, WINDOW_TOPLEVEL )

        self.theToolbar = GtkToolbar( ORIENTATION_VERTICAL, TOOLBAR_BOTH )
        self.add( self.theToolbar )
        self.set_data('toolbar', self.theToolbar)


    def setPluginList( self, pluginlist ):

        aPluginNameList = []
        aIndicator = 0
        
        if not pluginlist:
            return
        
        for aModule in pluginlist.values():

            aModuleName = aModule.theName
            aButtonName = string.replace( aModuleName, 'Window', '' )
            aPluginNameList.append( aModuleName )

            aPixMap = GtkPixmap( self, os.path.join( aModule.theDirectoryName,\
                                                     aModuleName ) + '.xpm' )

            if aIndicator == 0:
                aIndicator = 1
                aFirstButtonObj = GtkRadioButton()
                aFirstButton = \
                             self.theToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
                                                      aFirstButtonObj, aButtonName,
                                                      '', '', aPixMap, None )
                self.set_data( aModuleName, aFirstButton )
            else :
                aButtonObj = GtkRadioButton( aFirstButtonObj )
                aButton = \
                        self.theToolbar.append_element( TOOLBAR_CHILD_RADIOBUTTON,
                                                 aButtonObj, aButtonName,
                                                 '', '', aPixMap, None )
                self.set_data( aModuleName, aButton )
            self.set_data( 'plugin_list' , aPluginNameList )
            aFirstButton.set_active( 1 )

    def getSelectedPluginName( self ):
        aPluginList = self.get_data( 'plugin_list' )
        for aPluginName in aPluginList :
            aButton = self.get_data( aPluginName )
            if aButton.get_active() :
                aSelectedPluginName = aPluginName
        return aSelectedPluginName
        
if __name__ == "__main__":

    def mainLoop():
        aPaletteWindow = PaletteWindow()
        gtk.mainloop()

    def main():
        mainLoop()

    main()
