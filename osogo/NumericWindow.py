#!/usr/bin/env python

import string

import gtk
import gnome.ui
import GDK
import libglade

from Window import *


class NumericWindow(Window):

    def __init__( self, gladefile ):

        self.theHandlerMap = {'input':           self.input}

        Window.__init__( self, gladefile )
        self.addHandlers( self.theHandlerMap )
        
    def setText( self, name , text ):
        obj = self.getWidget(name)
        obj.set_text(text)
        
    def setLabel(self,name,text):
        obj = self.getWidget(name)
        obj.set_label(text)

    def input( self,obj ):
        aNumberString =  obj.get_text()
        aNumber = string.atof( aNumberString )
        print aNumberString


if __name__ == "__main__":

    from main import *

    def main():
        systemPath = '/CELL/CYTOPLASM'
        ID = 'ATP'
        FQPI = systemPath + ':' + ID  
        propertyName = 'quantity'
        propertyValue = '0.00000'
        aWindow = NumericWindow( 'NumericWindow.glade' )
        aWindow.addHandler( 'gtk_main_quit', mainQuit )
        aWindow.setText("spinbutton1", propertyValue)
        aWindow.setText("label1",ID)  
        aWindow.setLabel("frame1", propertyName)
        mainLoop()


    main()

















