#!/usr/bin/env python


from Window import *
from main import *
from Plugin import *

class MainWindow(Window):

    def __init__( self ):

        self.thePluginWindowManager = PluginWindowManager()

        self.theHandlerMap = { 'on_exit_activate':   self.exit
                               }
        Window.__init__( self )
        self.addHandlers( self.theHandlerMap )

    def exit( self, obj ):
        mainQuit()

    def input( self,obj ):
        aNumberString =  obj.get_text()
        aNumber = int( aNumberString )
        self.theProgressBar.set_value( aNumber )



