#!/usr/bin/env python

import string

from PluginWindow import *
from ecssupport import *
import GTK

class SubstanceWindow(PluginWindow):

    def __init__( self, dirname,  data, pluginmanager, root=None ):
        
        # 0 : not fixed  1: fixed
        self.theFixFlug = 0

        PluginWindow.__init__( self, dirname, data, pluginmanager, root )

        self['toolbar1'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar1'].set_button_relief( GTK.RELIEF_HALF )
        self['toolbar2'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar2'].set_button_relief( GTK.RELIEF_HALF )
        self['toolbar3'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar3'].set_button_relief( GTK.RELIEF_HALF )
        self['toolbar4'].set_style( GTK.TOOLBAR_ICONS )
        self['toolbar4'].set_button_relief( GTK.RELIEF_HALF )        
        
        self.addHandlers( {'button_toggled': self.fix_mode,
                              'qty_increase_pressed': self.increaseQuantity, 
                              'qty_decrease_pressed': self.decreaseQuantity,
                              'concentration_increase_pressed': self.increaseConcentration,
                              'concentration_decrease_pressed': self.decreaseConcentration,
                             'input_quantity': self.inputQuantity,
                             'input_concentration': self.inputConcentration
                             } )

        self.initialize()
        
    def initialize( self ):
        
        self.theFPNQuantity = tuple( convertFullIDToFullPN( self.theFullID(), 'Quantity' ) )
        self.theFPNConcentration = tuple(convertFullIDToFullPN( self.theFullID(), 'Concentration' ))

        self.theType = str( self.theFullID()[TYPE] )
        self.theID   = str( self.theFullID()[ID] )
        self.thePath = str( self.theFullID()[SYSTEMPATH] )
        
        aFullIDString = createFullIDString( self.theFullID() )
        self["id_label"].set_text( aFullIDString )

        self.update()

    def update( self ):

        self.theQuantity = self.theDriver.getProperty( self.theFPNQuantity )[0]
        self['Quantity_entry'].set_text( str( self.theQuantity ) )

        self.theConcentration = self.theDriver.getProperty( self.theFPNConcentration )[0]
        self['Concentration_entry'].set_text( str( self.theConcentration ) )
    
    def fix_mode( self, a ) :

        self.theFixFlug = 1 - self.theFixFlug
        if self.theFixFlug == 0:
            print 'not fixed'
        else:
            print 'fixed'

    def inputQuantity( self, obj ):

        aNumberString = obj.get_text()
        self.theQuantity = string.atof( aNumberString )
        self.changeQuantity()

    def inputConcentration( self, obj ):
        
        aNumberString = obj.get_text()
        self.theConcentration = string.atof( aNumberString )
        self.changeConcentration()
    
    def increaseQuantity( self, button_object ):

        self.theQuantity *= 2.0
        self[ "Quantity_entry" ].set_text( str( self.theQuantity ) )
        self.changeQuantity()

    def increaseConcentration( self, button_object ):

        self.theConcentration *= 2.0
        self[ "Concentration_entry" ].set_text( str( self.theConcentration ) )
        self.changeConcentration()

    def decreaseQuantity( self, button_object ):

        self.theQuantity *= 0.5
        self[ "Quantity_entry" ].set_text( str( self.theQuantity ) )
        self.changeQuantity()

    def decreaseConcentration( self, button_object ):

        self.theConcentration *= 0.5
        self[ "Concentration_entry" ].set_text( str( self.theConcentration ) )
        self.changeConcentration()

    def changeQuantity( self ):
        
        value = (self.theQuantity,)
        self.theDriver.setProperty( self.theFPNQuantity, value )
        self.thePluginManager.updateAllPluginWindow()
        print self.theDriver.getProperty( self.theFPNQuantity )

    def changeConcentration( self ):
        
        value = (self.theConcentration,)
        self.theDriver.setProperty( self.theFPNConcentration, value )
        self.thePluginManager.updateAllPluginWindow()
        print self.theDriver.getProperty( self.theFPNConcentration )

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()



if __name__ == "__main__":

    class simulator:        
        dic={('Substance','/CELL/CYTOPLASM','ATP','Quantity') : (1950,),
             ('Substance','/CELL/CYTOPLASM','ATP','Concentration') : (0.353,),}
        def getProperty( self, fpn ):
            return simulator.dic[fpn]
        
        def setProperty( self, fpn, value ):
            simulator.dic[fpn] = value
            
            
    fpn = ('Substance','/CELL/CYTOPLASM','ATP','')


    def mainQuit( obj, data ):
        print obj,data
        gtk.mainquit()
         
    def mainLoop():
        # FIXME: should be a custom function
        
        gtk.mainloop()

    def main():
        aSubstanceWindow = SubstanceWindow( 'plugins', simulator(), [fpn,])
        aSubstanceWindow.addHandler( 'gtk_main_quit' , mainQuit )
        aSubstanceWindow.update()

        mainLoop()

    main()
    


