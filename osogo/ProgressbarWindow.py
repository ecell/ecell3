#!/usr/bin/env python2

import string

import gtk
import gnome.ui
import GDK
import libglade
import Numeric


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

        self.theHandlerMap = {
                              'input1': self.input,
                              'input2': self.input,
                              'button_press_event': self.button_press_event
                              }

        Window.__init__( self, gladefile )
        self.addHandlers( self.theHandlerMap )


    def button_press_event (self,name,obj):
        print 'MAX automatically changed'

        
    def setText( self, name , text ):
        obj = self.getWidget(name)
        obj.set_text(text)

    def setValue2( self, name ,value ):
        obj = self.getWidget( name )
        log = (int)(Numeric.log10(value))
        obj.set_value(log)
        self.keisan(value,log)

    def keisan(self,value,log):
        value = (int)(value / (float)(10**(log -1)))
        self.theProgressBar = self.getWidget( "progressbar1" )
        self.theProgressBar.set_value(value)
        
    def setLabel(self,name,text):
        obj = self.getWidget(name)
        obj.set_label(text)

    def getPercentage(self,name,value):
        obj = self.getWidget(name)
        obj.set_value(value)
        
    def input( self,obj ):
        aNumberString =  obj.get_text()
        aNumber = string.atof( aNumberString )
        self.theSpinButton = self.getWidget( "spinbutton1" )
        self.theSpinButton.set_value( aNumber )
        value = propertyValue1
        self.keisan(value,aNumber)


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
    aWindow = MainWindow( 'ProgressbarWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )
    aWindow.setText("label1",ID)
    aWindow.setValue2("spinbutton1",propertyValue1)
#    aWindow.setLabel("frame1", propertyName)
    mainLoop()

if __name__ == "__main__":
    propertyValue1 = 750.0000
    main()

















