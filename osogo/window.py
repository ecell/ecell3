#!/usr/bin/env python

import gtk
import gnome.ui
import GDK
import libglade

import string
import sys

class Window:

    def __init__( self, gladefile, root=None ):
        self.widgets = libglade.GladeXML( gladefile, root=root )
        self.theHandlerMap = {}
        self.addHandlers( self.theHandlerMap )
        self.addHandler('delete', self.deletedWindow)

    def deletedWindow(self, a, b, c):
        pass
        
    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )

    def setText( self, aWidgetName , text ):
        obj = self.getWidget(aWidgetName)
        obj.set_text(text)

    def getText( self, aWidgetName ):
        obj = self.getWidget(aWidgetName)
        return obj.get_text()
        
    def setValue( self, aWidgetName , value ):
        obj = self.getWidget(aWidgetName)
        obj.set_value(value)

    def getValue( self, aWidgetName ):
        obj = self.getWidget(aWidgetName)
        return obj.get_value()

    def setLabel(self,aWidgetName,text):
        obj = self.getWidget(aWidgetName)
        obj.set_label(text)

### used in test code mainly

    def mainQuit( self, obj, data ):
        gtk.mainquit()

class Interface(Window):

    def __init__(self, gladefile):
        Window.__init__(self, gladefile)
        self.addHandler( 'copy', copyFQPPList)
#        self.addHandler( 'paste', copyFQPPList)
#        self.addHandler( 'add', addFQPPList)

    def copyFQPPList():
        print 'copy'
        
#    def pasteFQPPList():
#    def addFQPPList():

# class InterfaceForFQPP    

def mainLoop():
    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":
    main()

















