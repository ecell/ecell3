#!/usr/bin/env python

import string

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
        
        self.theHandlerMap = {
            }
        
        Window.__init__( self, gladefile )
        
        self.addHandlers( self.theHandlerMap)
        
        self.thePropertyClist = self.getWidget( "clist1" )
        self.theTypeEntry = self.getWidget( "entry_TYPE" )
        self.theIDEntry = self.getWidget( "entry_ID" )
        self.thePathEntry = self.getWidget( "entry_PATH" )
        self.theNameEntry = self.getWidget( "entry_NAME" )
        
        
        
        #    def setName( self ):
        #        MainWindow.thePropertyEntity.set_text('EntytyName')
        #    setName( self )
        
        
        
    def update( self ):
        aPropertyList = list( tmpget( 'PropertyList' ) )
        bPropertyList = testdic['Name'][0]
        #        aFQPP = list( FQPP )
        
        # remove keyword
        aPropertyList = aPropertyList[1:] 
        # remove PropertyList itself
        aPropertyList.remove( 'PropertyList' )
        aPropertyList.remove( 'Name' )
        
        self.thePropertyClist.clear()
        #        self.thePropertyListA.clear()
        #        self.thePropertyListB.clear()
        #        self.thePropertyListC.clear()
        
        
        #        print FQPPList[0]
        
        #        aFQPP = getFQPP()
        
        self.theTypeEntry.set_text(toString(FQPPList.getType()))
        self.theIDEntry.set_text(toString(FQPPList.getID()))
        self.thePathEntry.set_text(toString(FQPPList.getSystemPath()))
        self.theNameEntry.set_text(bPropertyList)
        
        
        
        for x in aPropertyList:
            if (x=='Substrate' or x=='Product') :
                
                for y in x:

                    aValueList = list( tmpget( y ) )
                    #aValueList = tmpget( y )
                    #aValueList = tmpget( x )
                    
                    #            aName = aValueList[0]
                    #            aValueList = aValueList[1:]
                    
                    aValueList = map( toString, aValueList )
                    
                    self.thePropertyClist.append( aValueList) 
                    
                    
                    #            print aValueList
                    
                else:
                    aValueList = tmpget( x )
                    aValueList = map( toString, aValueList )
                    
                    self.thePropertyClist.append( aValueList )
                

def toString( object ):
    return str( object )
    
def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    # FIXME: should be a custom function
    gtk.mainloop()

def main():
    aMainWindow = MainWindow( 'PropertyWindow.glade' )
    aMainWindow.addHandler( 'gtk_main_quit', mainQuit )    
    aMainWindow.update()
    mainLoop()

#this data should be written in an other file.
#the beginning of data area
    
testdic={'PropertyList': ('PropertyList', 'Name', 'A','B','C','Substrate','Product'),
          'Name': ('MichaelisMentenReactor', ),
          'A': ('aaa', ) ,
          'B': (1.04E-3, ) ,
          'C': (41, ),
          'Substrate': ('Substance:/CELL/CYTOPLASM/ ATP',
                        'Substance:/CELL/CYTOPLASM/ ADP',
                        'Substance:/CELL/CYTOPLASM/ AMP',
                        ),

          'Product': ('Substance:/CELL/CYTOPLASM/ GLU',
                      'Substance:/CELL/CYTOPLASM/ LAC',
                       'Substance:/CELL/CYTOPLASM/ PYR',
                      )
          } 

#FQPPList = ['MichaMen','/CELL/CYTOPLASM','MichaelisMentenReactor']
FQPPList = propertyname.FullPropertyName('Reactor:/CELL/CYTOPLASM:MichaMen:PropertyName')



#the ending of data area


def getFQPP():
    return FQPPList
def tmpget( name ):
    aList = list(testdic[name])
    aList.insert( 0, name )
    return tuple( aList )

if __name__ == "__main__":

    main()
