#!/usr/bin/env python


from PluginWindow import *
from ecssupport import *
import GTK

class NumericWindow( PluginWindow ):

    def __init__( self, dirname,  data, pluginmanager ):

        PluginWindow.__init__( self, dirname, data, pluginmanager )

        self['toolbar5'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar5'].set_button_relief( GTK.RELIEF_HALF )

        self.addHandlers( { 'input_value'    :self.inputValue,
                            'increase_value' :self.increaseValue,
                            'decrease_value' :self.decreaseValue } )

        self.initialize()

    def initialize( self ):
        aString = str( self.theFullPN()[ID] )
        aString += ':\n' + str( self.theFullPN()[PROPERTY] )
        self["id_label"].set_text( aString )
        self.update()
        
    def update( self ):
        self["value_frame"].set_text( str( self.getValue() ) )

    def inputValue( self, obj ):
        aValue =  string.atof( obj.get_text() )
        self.setValue( aValue )

    def increaseValue( self, obj ):
        self.setValue( self.getValue() * 2.0 )

    def decreaseValue( self, obj ):
        self.setValue( self.getValue() * 0.5 )

    def getValue( self ):
        aValueList = self.theSimulator.getEntityProperty( self.theFullPN() )
        return aValueList[0]

    def setValue( self, aValue ):
        aValueList = ( aValue, )
        self.theSimulator.setEntityProperty( self.theFullPN(), aValueList )
        self.thePluginManager.updateAllPluginWindow()

### test code

if __name__ == "__main__":

    class simulator:

        dic={('Variable', '/CELL/CYTOPLASM', 'ATP','Value') : (1950,),}

        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]

        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Variable','/CELL/CYTOPLASM','ATP','Value')

    def mainQuit( obj, data ):
        print obj,data
        gtk.main_quit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.main()

    def main():
        aNumericWindow = NumericWindow( 'plugins', simulator(), [fpn,] )
        aNumericWindow.addHandler( 'gtk_main_quit', mainQuit )
        aNumericWindow.update()

        mainLoop()

    main()









