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

        theHandlerMap = { 'on_label4_key_press_event': self.button1_clicked,
                          'input': self.input
                          }

        Window.__init__( self, gladefile )
        self.addHandlers( theHandlerMap )

        self.theProgressBar = self.getWidget( "progressbar1" )
        self.theProperty_value = self.getWidget( "clist1" )
        self.theProperty_entity= self.getWidget("label3")

    def button1_clicked( self,obj ):
        print 'hello'
        
    def input( self,obj ):
        aNumberString =  obj.get_text()
        aNumber = string.atoi( aNumberString )
        self.theProgressBar.set_value( aNumber )
        

def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()
    
def main():
    aWindow = MainWindow( 'property_window.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )
    aWindow.theProperty_value.set_column_width(0,100)
    for x in testlist:
        aWindow.theProperty_value.append(x)
#    aWindow.theProperty_value.append( ( 'property1' , 'value1') )
#    aWindow.theProperty_value.append( ( 'property2' , 'value2') )
    aWindow.theProperty_entity.set_text('Reactor')
    mainLoop()
    
if __name__ == "__main__":
    testlist = ( ('A',''),('B','' ),('C','') )
    main()
    
        

























