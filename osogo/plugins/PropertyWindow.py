#!/usr/bin/env python


import string

from PluginWindow import *
from ecssupport import *

class PropertyWindow(PluginWindow):

    
    def __init__( self, dirname, sim, data ):
        
        PluginWindow.__init__( self, dirname, sim, data )
        
        self.addHandlers( { 'input_row_pressed'   : self.select_property,
                            'show_button_pressed' : self.show } )
        
        self.thePropertyClist = self.getWidget( "clist1" )
        self.theTypeEntry     = self.getWidget( "entry_TYPE" )
        self.theIDEntry       = self.getWidget( "entry_ID" )
        self.thePathEntry     = self.getWidget( "entry_PATH" )
        self.theClassNameEntry     = self.getWidget( "entry_NAME" )
        
        self.initialize()

        
    def initialize( self ):

        self.theSelected = ''
        
        self.theFullID = FullPropertyNameToFullID( self.theFPNs[0] )
        self.theType = PrimitiveTypeString[ self.theFullID[TYPE] ]
        self.theID   = str( self.theFullID[ID] )
        self.thePath = str( self.theFullID[SYSTEMPATH] )
        aFullPropertyName = FullIDToFullPropertyName( self.theFullID,
                                                      'ClassName' )
        aList = self.theSimulator.getProperty( aFullPropertyName )
        self.theClassName  = aList[0]
        self.theTypeEntry.set_text( self.theType  )
        self.theIDEntry.set_text  ( self.theID )
        self.thePathEntry.set_text( self.thePath )
        self.theClassNameEntry.set_text( self.theClassName )

        self.update()
        

    def update( self ):

        self.updatePropertyList()
        self.thePropertyClist.clear()
        for aValue in self.theList:
            self.thePropertyClist.append( aValue )

        
    def updatePropertyList( self ):

        self.theList = []

        aFullPropertyName = FullIDToFullPropertyName( self.theFullID,
                                                      'PropertyList' )
        aPropertyList =\
        list( self.theSimulator.getProperty( aFullPropertyName ) )

        
        # remove PropertyList and ClassName
        aPropertyList.remove( 'PropertyList' )
        aPropertyList.remove( 'ClassName' )

        for aProperty in aPropertyList:

            aFullPropertyName = FullIDToFullPropertyName( self.theFullID,
                                                          aProperty )
            aValueList = self.theSimulator.getProperty( aFullPropertyName ) 

            aLength = len( aValueList )
            if  aLength > 1 :
                aNumber = 1
                for aValue in aValueList :
                    aList = [ aProperty, aNumber, aValue ]
                    aList = map( str, aList )
                    self.theList.append( aList ) 
                    aNumber += 1
            else:
                for aValue in aValueList :
                    aList = [ aProperty, '', aValue ]
                    aList = map( str, aList )
                    self.theList.append( aList )

    def select_property(self, obj, data1, data2, data3):

        aSelectedItem = self.theList[data1]
        aFullPropertyName = None

        print aSelectedItem
        try:
            aFullPropertyName = FullPropertyName( aSelectedItem[2] )
        except ValueError:
            pass

        if not aFullPropertyName:
            try:
                aFullID = FullID( aSelectedItem[2] )
                aFullPropertyName = FullIDToFullPropertyName( aFullID )
            except ValueError:
                pass
            
        if not aFullPropertyName:
            aFullPropertyName = [ self.theType, self.thePath,
                          self.theID, aSelectedItem[0] ]

        self.theSelected = aFullPropertyName

    def show( self, obj ):

        print self.theSelected



if __name__ == "__main__":


    class simulator:

        dic={'PropertyList':
             ('PropertyList', 'ClassName', 'A','B','C','Substrate','Product'),
             'ClassName': ('MichaelisMentenReactor', ),
             'A': ('aaa', ) ,
             'B': (1.04E-3, ) ,
             'C': (41, ),
             'Substrate': ('Substance:/CELL/CYTOPLASM:ATP',
                           'Substance:/CELL/CYTOPLASM:ADP',
                           'Substance:/CELL/CYTOPLASM:AMP',
                           ),
             'Product': ('Substance:/CELL/CYTOPLASM:GLU',
                         'Substance:/CELL/CYTOPLASM:LAC',
                         'Substance:/CELL/CYTOPLASM:PYR',
                         )
             } 

        def getProperty( self, fpn ):
            return simulator.dic[fpn[PROPERTY]]
    
    fpn = FullPropertyName('Reactor:/CELL/CYTOPLASM:MichaMen:PropertyName')



    def mainQuit( obj, data ):
        print obj,data
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aPropertyWindow = PropertyWindow( 'plugins', simulator(), [fpn,] )
        aPropertyWindow.addHandler( 'gtk_main_quit', mainQuit )
        aPropertyWindow.update()

        mainLoop()

    

    main()











