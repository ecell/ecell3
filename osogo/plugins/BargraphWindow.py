#!/usr/bin/env python2

import string

### for test
import sys
sys.path.append('.')
import Plugin
### for test

from PluginWindow import *
from ecssupport import *

import Numeric
import GTK


class BargraphWindow(PluginWindow):
    
    def __init__( self, dirname, sim, data, pluginmanager ):

        PluginWindow.__init__( self, dirname, sim, data, pluginmanager )

        #test
        self['toolbar5'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar6'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar5'].set_button_relief( GTK.RELIEF_HALF )
        self['toolbar6'].set_button_relief( GTK.RELIEF_HALF )        

        self.pull = 0
        self.thePositiveFlag = 1
        self.theAutoChangeFlag = 1
        self.theActualValue = 0
        self.theBarLength = 0
        self.theMultiplier = 0
        
        self.addHandlers( { \
            'on_add_button_clicked' : self.updateByAddbutton,
            'on_subtract_button_clicked' : self.updateBySubtractbutton,
            'multiplier_entry_activate' : self.updateByTextentry,
            'auto_button_toggled': self.updateByAutoButton ,
            })

        self.theIDEntry = self.getWidget( "property_id_label" )
        self.theMultiplier1Entry = self.getWidget("multiplier1_label")

        aPropertyListFullPN = convertFullIDToFullPN(self.theFullID(),
                                                  'PropertyList')
        aPropertyList =\
        list( self.theSimulator.getProperty( aPropertyListFullPN ) )
        aAttributeListFullPN = convertFullIDToFullPN(self.theFullID(),
                                                  'PropertyAttributes')
        aAttributeList =\
        list(self.theSimulator.getProperty( aAttributeListFullPN ))
        num = 0

        for aProperty in aPropertyList:
            if (aProperty == 'Quantity'):
                print aProperty,
                print "=",
                print aAttributeList[num]
            else :
                pass
            num += 1
        self.initialize()
        
    def initialize( self ):

        self.theSelected = ''
        
        self.theType = str( self.theFullID()[TYPE] )
        self.theID   = str( self.theFullID()[ID] )
        self.thePath = str( self.theFullID()[SYSTEMPATH] )

        self.update()
        
    def update( self ):
        aString = str( self.theFullPN()[ID] )
        aString += ':\n' + str( self.theFullPN()[PROPERTY] )        
        self.theIDEntry.set_text  ( aString )

        aValue = self.theSimulator.getProperty( self.theFullPN() )
        value = aValue[0]
        self.theActualValue = value
        self.theBarLength , self.theMultiplier , self.thePositiveFlag \
                          = self.calculateBarLength( value )

        aIndicator = (value / (float)(10**(self.theMultiplier))) \
                     * self.thePositiveFlag


        self['progressbar'].set_value(int(self.theBarLength))
        self['progressbar'].set_format_string(str(value))

        self.theMultiplier1Entry.set_text(str(int(self.theMultiplier-1)))
        self['multiplier_entry'].set_text(str(int(self.theMultiplier+2)))

    def updateByAuto( self, value ):
        self.theAutoChangeFlag = 1
        self.update()

    def updateByAddbutton( self , obj ):
        self['auto_button'].set_active( 0 )
        aNumberString =  self['multiplier_entry'].get_text()
        aNumber = string.atof( aNumberString )
        aNumber = aNumber + 1
        self.pull = aNumber

        self.theAutoChangeFlag = 0
        self.update()

    def updateBySubtractbutton( self,obj ):
#        if self.theAutoChangeFlag :
#            pass
#        else :
        self['auto_button'].set_active( 0 )

        aNumberString =  self['multiplier_entry'].get_text()
        aNumber = string.atof( aNumberString )
        aNumber = aNumber - 1
        self.pull = aNumber

        self.theAutoChangeFlag = 0
        self.update()

    def updateByTextentry(self, obj):

        if self.theAutoChangeFlag :
            pass
        else :
            self['auto_button'].set_active( 0 )

        aNumberString = obj.get_text()

        aNumber = string.atof( aNumberString )
        self.pull = aNumber

        self.theAutoChangeFlag = 0
        self.update()

    def updateByAutoButton(self, autobutton):
        self.update()

    def calculateBarLength( self, value ):
        if value < 0 :
            value = - value
            aPositiveFlag = -1
        else :
            aPositiveFlag = 1

        if self['auto_button'].get_active() :
            if value == 0 :
                aMultiplier = 2
            else :
                aMultiplier = (int)(Numeric.log10(value))
            self.pull = aMultiplier+2
        else :
            aMultiplier = self.pull-2

        if value == 0:
            aBarLength = 0
        else :
            aBarLength = (Numeric.log10(value)+1-aMultiplier)*1000/3

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
        aPluginManager = Plugin.PluginManager()
        aBargraphWindow = BargraphWindow( 'plugins', simulator(), [fpn,], aPluginManager )
        aBargraphWindow.addHandler( 'gtk_main_quit', mainQuit )
        
        mainLoop()


    # propertyValue1 = -750.0000

    main()
