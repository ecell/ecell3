#!/usr/bin/env python2

import string

from Plugin import *
from ecssupport import *

import Numeric

class ProgressbarWindow(PluginWindow):
    
    def __init__( self, dirname, sim, data ):

        PluginWindow.__init__( self, dirname, sim, data )
        self.thePositiveFlag = 1
        self.theAutoChangeFlag = 1
        self.theActualValue = 0
        self.theBarLength = 0
        self.theMultiplier = 0
        
        self.addHandlers( { \
            'level_spinbutton_activate': self.updateBySpinbutton,
            'level_spinbutton_changed': self.updateBySpinbutton,
            'auto_button_toggled': self.updateByAutoButton ,
#            'on_entry1_activate': self.changeValueFromEntryWindow
            })
        # self['property_id_label'].set_text(self.theID)
        self.theIDEntry = self.getWidget( "property_id_label" )
        self.theFPN = data[0]
        self.initialize()
        
    def initialize( self ):

        self.theSelected = ''
        
        self.theFullID = convertToFullID( self.theFPN )
        self.theType = str( self.theFullID[TYPE] )
        self.theID   = str( self.theFullID[ID] )
        self.thePath = str( self.theFullID[SYSTEMPATH] )

        self.update()
        
    def update( self ):
        self.theIDEntry.set_text  ( self.theID )
        aValue = self.theSimulator.getProperty( self.theFPN )
        value = aValue[0]
#        value = 124143.807
        self.theActualValue = value
        self.theBarLength , self.theMultiplier , self.thePositiveFlag \
                          = self.calculateBarLength( value )

        # aIndicator = self.theBarLength * self.thePositiveFlag
        aIndicator = (value / (float)(10**(self.theMultiplier))) \
                     * self.thePositiveFlag
        
        self['progressbar'].set_value(int(self.theBarLength))
        self['progressbar'].set_format_string(str(aIndicator))
        self['level_spinbutton'].set_value(self.theMultiplier)

    def updateByAuto( self, value ):
        self.theAutoChangeFlag = 1
        self.update()

    def updateBySpinbutton( self, spinbutton_obj ):
        if self.theAutoChangeFlag :
            pass
        else :
            self['auto_button'].set_active( 0 )
        self.update()

        aNumberString =  spinbutton_obj.get_text()
        aNumber = string.atof( aNumberString )
        self['level_spinbutton'].set_value(aNumber)

        self.theAutoChangeFlag = 0

    def updateByAutoButton(self, autobutton):
        self.update()

    def calculateBarLength( self, value ):
        if value < 0 :
            value = - value
            aPositiveFlag = -1
        else :
            aPositiveFlag = 1

        if self['auto_button'].get_active() :
            aMultiplier = (int)(Numeric.log10(value))
        else :
            aMultiplier = self['level_spinbutton'].get_value()

        # aBarLength = (value / (float)(10**(aMultiplier)))
        aBarLength = (Numeric.log10(value)-aMultiplier+1)*10/3

        return  aBarLength, aMultiplier, aPositiveFlag
                
    ####### for test #############################
    #def changeValueFromEntryWindow( self, obj, a):
    #    
    #    aValueString = obj.get_text()
    #    aValue = string.atof( aValueString )
    #    print aValue
    #    self.changeValue( aValue )
    #
    #############################################

    def changeValue( self, value ):
        self.updateByAuto( value )
    

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
        aProgressbarWindow = ProgressbarWindow( 'plugins', simulator(), [fpn,] )
        aProgressbarWindow.addHandler( 'gtk_main_quit', mainQuit )
        
        mainLoop()


    # propertyValue1 = -750.0000
    main()
