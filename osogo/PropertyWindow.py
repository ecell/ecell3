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
        
        
        
        
        
    def update( self, aMainValueList, namePropertyList ):
        self.thePropertyClist.clear()
        
        
        self.theTypeEntry.set_text( toString( FQPPList.getType() ) )
        self.theIDEntry.set_text( toString( FQPPList.getID() ) )
        self.thePathEntry.set_text( toString( FQPPList.getSystemPath() ) )
        self.theNameEntry.set_text( namePropertyList )
        for a in aMainValueList:
            self.thePropertyClist.append( a )


        
    def makeValueList( self ):
        self.MainValueList = []
        

        aPropertyList = list( getKeyandList( 'PropertyList' ) )
        self.bPropertyList = testdic['Name'][0] 
        
        
        # remove keyword
        aPropertyList = aPropertyList[1:] 
        # remove PropertyList itself
        aPropertyList.remove( 'PropertyList' )
        aPropertyList.remove( 'Name' )

        
        
        for x in aPropertyList:
            aValueList = getKeyandList( x )
            aValueList = aValueList[1:]
            num_list = len( aValueList )
            r=1
            if  num_list > 1 :
                for y in aValueList :
                    
                    aSubstanceList = [ x,r,y ]
                    aSubstanceList = map( toString, aSubstanceList )
                    self.MainValueList.append( aSubstanceList ) 
                    r=r+1
                
                
            else:
                for y in aValueList :
                    aParameterList = [ x,'',y ]
                    aParameterList = map( toString, aParameterList )
                    self.MainValueList.append( aParameterList ) 
                    
        return self.MainValueList, self.bPropertyList
            

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
    aMainWindow.makeValueList()
    aMainWindow.update( aMainWindow.MainValueList, aMainWindow.bPropertyList )
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


#def getFQPP():
#    return FQPPList

def getKeyandList( key ):
    aList = list(testdic[key])
    aList.insert( 0, key )
    return tuple( aList )

if __name__ == "__main__":

    main()











