#!/usr/bin/env python


from PluginWindow import *
from ecssupport import *
import GTK

class DigitalWindow( PluginWindow ):

    def __init__( self, dirname,  data, pluginmanager, root=None ):

        PluginWindow.__init__( self, dirname, data, pluginmanager, root )

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
        aValueList = self.theSimulator.getProperty( self.theFullPN() )
        return aValueList[0]

    def setValue( self, aValue ):
        aValueList = ( aValue, )
        self.theSimulator.setProperty( self.theFullPN(), aValueList )
        self.thePluginManager.updateAllPluginWindow()

### test code

if __name__ == "__main__":

    class simulator:

        dic={('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

        def getProperty( self, fpn ):
            return simulator.dic[fpn]

        def setProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Substance','/CELL/CYTOPLASM','ATP','Quantity')

    def mainQuit( obj, data ):
        print obj,data
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aDigitalWindow = DigitalWindow( 'plugins', simulator(), [fpn,] )
        aDigitalWindow.addHandler( 'gtk_main_quit', mainQuit )
        aDigitalWindow.update()

        mainLoop()

    main()









