#!/usr/bin/env python

import string
import ecs
import gtk
import gnome.ui
import GDK
import libglade
import propertyname

class Window:

    def __init__( self, gladefile=None, root=None ):
        self.widgets = libglade.GladeXML( filename=gladefile, root=root )

    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )

class MainWindow(Window):

    def __init__( self, gladefile ):

        self.theHandlerMap = {'button_toggled': self.fix_mode,
                              'qty_increase_pressed': self.increaseQuantity, 
                              'qty_decrease_pressed': self.decreaseQuantity ,
                              'concentration_increase_pressed': self.increaseConcentration,
                              'concentration_decrease_pressed': self.decreaseConcentration,                              }
        
        Window.__init__( self, gladefile )
        self['label86'].set_text( str( theFullID ) )
        self['Quantity_entry'].set_text( str( theQuantity ) )
        self['Concentration_entry'].set_text( str( theConcentration ) )

        self.addHandlers( self.theHandlerMap )

    def fix_mode( self, a ) :
        print 'hello'

    def increaseConcentration( self, button_object ):
        self.increaseValue('Concentration_entry')

    def increaseQuantity( self, button_object ):
        self.increaseValue('Quantity_entry')

    def decreaseConcentration( self, button_object ):
        self.decreaseValue('Concentration_entry')

    def decreaseQuantity( self, button_object ):
        self.decreaseValue('Quantity_entry')

    def increaseValue( self, name ):
        aValue = self[ name ].get_text()
        aNewValue = string.atof(aValue) * 2
        self[ name ].set_text( str( aNewValue ) )

    def decreaseValue( self, name ):
        aValue = self[ name ].get_text()
        aNewValue = string.atof(aValue) / 2
        self[ name ].set_text( str( aNewValue ) )

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()

def main():
    aWindow = MainWindow( 'SubstanceWindow.glade' )


    mainLoop()

if __name__ == "__main__":

    theQuantity = 100
    theConcentration = 0.0145
    theVolume = 1e-18
    theFullID = 'Substance:/CELL/CYTOPLASM:ATP'
    theAvogadoroNumber = 6.022 * 1e-23

    main()
    


