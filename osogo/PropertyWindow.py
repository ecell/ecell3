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

        self.theHandlerMap = {
            }
        
        Window.__init__( self, gladefile )

        self.addHandlers( self.theHandlerMap)
         
        self.thePropertyClist = self.getWidget( "clist1" )
        self.thePropertyName= self.getWidget( "label3" )


        #    def setName( self ):
        #        MainWindow.thePropertyEntity.set_text('EntytyName')
        #    setName( self )
       
    def update( self ):
        aPropertyList = list( tmpget( 'PropertyList' ) )

        # remove keyword
        aPropertyList = aPropertyList[1:] 
        # remove PropertyList itself
        aPropertyList.remove( 'PropertyList' )

        self.thePropertyClist.clear()

        for x in aPropertyList:
            aValueList = tmpget( x )
            #            aName = aValueList[0]
            #            aValueList = aValueList[1:]
            aValueList = map( toString, aValueList )

            self.thePropertyClist.append( aValueList )
            print aValueList


def toString( object ):
    return str( object )
    
def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()

def main():
    aMainWindow = MainWindow( 'property_window.glade' )
    aMainWindow.addHandler( 'gtk_main_quit', mainQuit )    
    aMainWindow.update()
    mainLoop()
    
testdic={ 'PropertyList': ('PropertyList', 'A','B','C'),
          'A': ('aaa', ) ,'B': (1.04E-3, ) ,'C': (41, ) }

def tmpget( name ):
    aList = list(testdic[name])
    aList.insert( 0, name )
    return tuple( aList )

if __name__ == "__main__":

    main()
