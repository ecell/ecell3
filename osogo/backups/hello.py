#!/usr/bin/env python

import string

import gtk
import gnome.ui
import GDK
import libglade

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

        theHandlerMap = { 'on_button1_clicked': self.button1_clicked }

        Window.__init__( self, gladefile )
        self.addHandlers( theHandlerMap )

    def button1_clicked( self,obj ):
        print 'hello!'

def mainLoop():
    # FIXME: should be a custom function
    gtk.main()

def main():
    aWindow = MainWindow( 'hello.glade' )
    mainLoop()

if __name__ == "__main__":
    main()

