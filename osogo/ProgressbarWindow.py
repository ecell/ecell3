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

        self.theHandlerMap = {'input':           self.input,
                              'button_press_event': self.button_press_event
                              }

        Window.__init__( self, gladefile )
        self.addHandlers( self.theHandlerMap )


    def button_press_event (self,name,obj):
        print 'MAX automatically changed'

        
    def setText( self, name , text ):
        obj = self.getWidget(name)
        obj.set_text(text)

    def setValue( self, name ,value ):
        obj = self.getWidget( name )
        obj.set_value(value)
        
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
#    ID = 'ATPase: Activity'
    ID = 'ATPase'
    FQPI = systemPath + ':' + ID  
#    propertyName = 'quantity'
    propertyValue = 50.0000
    aWindow = MainWindow( 'ProgressbarWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )
    aWindow.setValue("progressbar1", propertyValue)
    aWindow.setText("label1",ID)  
#    aWindow.setLabel("frame1", propertyName)
    mainLoop()

if __name__ == "__main__":
    main()

















