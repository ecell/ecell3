#!/usr/bin/env python


from PluginWindow import *
from ecssupport import *

class NumericWindow( PluginWindow ):

    def __init__( self, dirname, sim, data, pluginmanager ):

        PluginWindow.__init__( self, dirname, sim, data, pluginmanager )

        self.addHandlers( { 'input_value'    :self.inputValue,
                            'increase_value' :self.increaseValue,
                            'decrease_value' :self.decreaseValue } )

        self.theFullPN = data[0]
        self.initialize(self.theFullPN)
        self.thePluginManager = pluginmanager

    def initialize( self, fpn ):

        self.theFullPN = fpn
        self.theID = str( self.theFullPN[ID] )
        
        self["id_label"].set_text( self.theID )
        value = self.theSimulator.getProperty( self.theFullPN )
        self.theCurValue = value[0]
        self["value_frame"].set_text(str(self.theCurValue))

        
    def update( self ):

        value = self.theSimulator.getProperty( self.theFullPN )
        self.theCurValue = value[0]
        self["value_frame"].set_text(str(self.theCurValue))

    def inputValue( self, obj ):

        aNumberString =  obj.get_text()
        self.theCurValue = string.atof( aNumberString )
        self.changeValue()

    def increaseValue( self, value ):

        self.theCurValue *= 2.0
        self["value_frame"].set_text(str(self.theCurValue))
        self.changeValue()

    def decreaseValue( self, obj ):

        self.theCurValue *= 0.5
        self["value_frame"].set_text(str(self.theCurValue))
        self.changeValue()

    def changeValue( self ):

        value = (self.theCurValue,)
        self.theSimulator.setProperty(self.theFullPN, value)
        self.thePluginManager.updateAllPluginWindow()

        ### for check
        self.thePluginManager.printMessage( getFullPNString( self.theFullPN ) )
        self.thePluginManager.printMessage( ' -> ' )
        self.thePluginManager.printMessage( str( self.theSimulator.getProperty(self.theFullPN)[0] ) )
        self.thePluginManager.printMessage( "\n" )

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
        aNumericWindow = NumericWindow( 'plugins', simulator(), [fpn,] )
        aNumericWindow.addHandler( 'gtk_main_quit', mainQuit )
        aNumericWindow.update()

        mainLoop()

    main()









