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

#    def label1_enter_notify_event( self,obj ):
#        print 'zhaiteng'
        
def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()

def main():
    systemPath = '/CELL/CYTOPLASM'
    ID = 'ATP'
    FQPI = systemPath + ':' + ID  
    propertyName = 'quantity'
    propertyValue = '0.00000'
    aWindow = MainWindow( 'NumericWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )
    aWindow.setText("spinbutton1", propertyValue)
    aWindow.setText("label1",ID)  
    aWindow.setLabel("frame1", propertyName)
    mainLoop()

if __name__ == "__main__":
    main()

















