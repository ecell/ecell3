#!/usr/bin/env python


from Plugin import *
from ecssupport import *

class NumericWindow( PluginWindow ):

    def __init__( self, dirname, sim, data ):

        PluginWindow.__init__( self, dirname, sim, data )

        self.addHandlers( { 'input_value'    :self.inputValue,
                            'increase_value' :self.increaseValue,
                            'decrease_value' :self.decreaseValue } )

        self.theFPN = data
        self.initInterface(self.theFPN)
        

    def initialize( self ):

        self.theFullID = convertToFullID( self.theFPN )
        self.theType = str( self.theFullID[TYPE] )
        self.theID   = str( self.theFullID[ID] )
        self.thePath = str( self.theFullID[SYSTEMPATH] )
        
#        self.update()

    def initInterface( self, fpn ):

        self.theFPN = fpn
        self.initialize()
        self["id_label"].set_text( self.theID )
        value = self.theSimulator.getProperty( self.theFPN )
        self.theCurValue = value[0]
        self["value_frame"].set_text(str(self.theCurValue))

        
    def update( self ):

        pass


    def inputValue( self, obj ):

        aNumberString =  obj.get_text()
        self.theCurValue = string.atof( aNumberString )
        self.changeValue()

    def increaseValue( self, value ):

        self.theCurValue = self.theCurValue * 2.0
        self["value_frame"].set_text(str(self.theCurValue))
        self.changeValue()

    def decreaseValue( self, obj ):

        self.theCurValue = self.theCurValue * 0.5
        self["value_frame"].set_text(str(self.theCurValue))
        self.changeValue()

    def changeValue( self ):

        value = (self.theCurValue,)
        self.theSimulator.setProperty(self.theFPN, value)
        print self.theSimulator.getProperty(self.theFPN)


### test code

if __name__ == "__main__":

    class simulator:

        dic={('Substance','/CELL/CYTOPLASM','ATP','quantity') : (1950,),}

        def getProperty( self, fpn ):
            return simulator.dic[fpn]

        def setProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Substance','/CELL/CYTOPLASM','ATP','quantity')

    def mainQuit( obj, data ):
        print obj,data
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aNumericWindow = NumericWindow( 'plugins', simulator(), fpn )
        aNumericWindow.addHandler( 'gtk_main_quit', mainQuit )
        aNumericWindow.update()

        mainLoop()

    main()









