#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

import string

### for test
import sys
sys.path.append('.')
import Plugin
### for test

from PluginWindow import *
from ecssupport import *

import numpy
import GTK


class ProgressbarWindow(PluginWindow):
    
    def __init__( self, dirname, sim, data, pluginmanager ):

        PluginWindow.__init__( self, dirname, sim, data, pluginmanager )

        #test
        self['toolbar3'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar4'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar3'].set_button_relief( GTK.RELIEF_HALF )
        self['toolbar4'].set_button_relief( GTK.RELIEF_HALF )        

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
        self.theMultiplier2Entry = self.getWidget("multiplier2_label")
        self.theMultiplier3Entry = self.getWidget("multiplier3_label")
        self.theFPN = data[0]
        self.theFullID = convertFullPNToFullID( self.theFPN )
        aFullPropertyName = convertFullIDToFullPN(self.theFullID,
                                                  'PropertyList')
        aPropertyList =\
        list( self.theSimulator.getEntityProperty( aFullPropertyName ) )
        aAttributeList = convertFullIDToFullPN(self.theFullID,
                                                  'PropertyAttributes')
        aAttributeList =\
        list(self.theSimulator.getEntityProperty( aAttributeList ))
        num = 0

        for aProperty in aPropertyList:
            if (aProperty == 'Value'):
                print aProperty,
                print "=",
                print aAttributeList[num]
            else :
                pass
            num += 1
        self.initialize()
        
    def initialize( self ):

        self.theSelected = ''
        
        self.theType = str( self.theFullID[TYPE] )
        self.theID   = str( self.theFullID[ID] )
        self.thePath = str( self.theFullID[SYSTEMPATH] )

        self.update()
        
    def update( self ):
        self.theIDEntry.set_text  ( self.theID )
        
        aValue = self.theSimulator.getEntityProperty( self.theFPN )
        value = aValue[0]
        self.theActualValue = value
        self.theBarLength , self.theMultiplier , self.thePositiveFlag \
                          = self.calculateBarLength( value )

        # aIndicator = self.theBarLength * self.thePositiveFlag
        aIndicator = (value / (float)(10**(self.theMultiplier))) \
                     * self.thePositiveFlag

#        print self.theMultiplier
        
        self['progressbar'].set_value(int(self.theBarLength))
        self['progressbar'].set_format_string(str(value))

        self.theMultiplier1Entry.set_text(str(int(self.theMultiplier-1)))
        self.theMultiplier2Entry.set_text(str(int(self.theMultiplier)))
        self.theMultiplier3Entry.set_text(str(int(self.theMultiplier+1)))
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

#        if value == 0 :
#            aMultiplier = 0
#            aBarLength = 0

#        else :
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

        dic={('Variable','/CELL/CYTOPLASM','ATP','value') : (1950,),}
        
        def getEntityProperty( self, fpn ):
            return simulator.dic[fpn]
        
        def setEntityProperty( self, fpn, value ):
            simulator.dic[fpn] = value


    fpn = ('Variable','/CELL/CYTOPLASM','ATP','value')

    def mainQuit( obj, data ):
        print obj,data
        gtk.main_quit()

    def mainLoop():
        # FIXME: should be a custom function
        gtk.main()

    def main():
        aPluginManager = Plugin.PluginManager()
        aProgressbarWindow = ProgressbarWindow( 'plugins', simulator(), [fpn,], aPluginManager )
        aProgressbarWindow.addHandler( 'gtk_main_quit', mainQuit )
        
        mainLoop()


    # propertyValue1 = -750.0000

    main()
