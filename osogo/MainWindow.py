#!/usr/bin/env python


from Window import *
from main import *

class MainWindow(Window):

    def __init__( self ):

        self.theHandlerMap = { 'on_exit_activate':   self.exit
                               }

        Window.__init__( self )
        self.addHandlers( self.theHandlerMap )

        self.theProgressBar = self.getWidget( "progressbar1" )

    def exit( self, obj ):
        mainQuit()

    def input( self,obj ):
        aNumberString =  obj.get_text()
        aNumber = int( aNumberString )
        self.theProgressBar.set_value( aNumber )



