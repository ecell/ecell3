#!/usr/bin/env python


import string

### for test
import sys
sys.path.append('.')
import Plugin
### for test


from PluginWindow import *
from ecssupport import *

class PropertyWindow(PluginWindow):

    
    def __init__( self, dirname, sim, data, pluginmanager ):
        
        PluginWindow.__init__( self, dirname, sim, data, pluginmanager )
        
        self.addHandlers( { 'input_row_pressed'   : self.selectProperty,
                            'show_button_pressed' : self.show } )
        
        self.thePropertyClist = self.getWidget( "property_clist" )
        self.theTypeEntry     = self.getWidget( "entry_TYPE" )
        self.theIDEntry       = self.getWidget( "entry_ID" )
        self.thePathEntry     = self.getWidget( "entry_PATH" )
        self.theClassNameEntry     = self.getWidget( "entry_NAME" )
        
        self.initialize()

        
    def initialize( self ):

        if self.theRawFullPNList == ():
            return
        self.setFullPNList()
    
    def setFullPNList( self ):
        self.theSelected = ''
        
        aNameFullPN = convertFullIDToFullPN( self.theFullID(),
                                                      'Name' )
        aNameList = list( self.theSimulator.getProperty( aNameFullPN ) )
        self.theClassName = aNameList[0]
        
        self.theType = PrimitiveTypeString[ self.theFullID()[TYPE] ]
        self.theID   = str( self.theFullID()[ID] )
        self.thePath = str( self.theFullID()[SYSTEMPATH] )
        aClassNameFullPN = convertFullIDToFullPN( self.theFullID(),
                                                  'ClassName' )
        aList = self.theSimulator.getProperty( aClassNameFullPN )

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

        aPropertyListFullPN = convertFullIDToFullPN( self.theFullID(),
                                                     'PropertyList' )
        aPropertyList =\
               list( self.theSimulator.getProperty( aPropertyListFullPN ) )

        aAttributeList = convertFullIDToFullPN(self.theFullID(),
                                                  'PropertyAttributes')
        aAttributeList =\
        list(self.theSimulator.getProperty( aAttributeList ))
        num = 0
 
        for aProperty in aPropertyList:
            if (aProperty == 'ClassName'):
                pass
            elif (aProperty == 'PropertyList'):
                pass
            elif (aProperty == 'PropertyAttributes'):
                pass
            elif (aProperty == 'FullID'):
                pass
            elif (aProperty == 'ID'):
                pass
            elif (aProperty == 'Name'):
                pass

            else :
                
                aFullPN = convertFullIDToFullPN( self.theFullID(),
                                                          aProperty )
            
                aValueList = self.theSimulator.getProperty( aFullPN ) 
                aLength = len( aValueList )
                aAttribute = self.getAttribute( aFullPN )
            
                if  aLength > 1 :
                    aNumber = 1
                    for aValue in aValueList :
                        aList = [ aProperty, aNumber, aValue , aAttribute ]
                        aList = map( str, aList )
                        self.theList.append( aList ) 
                        aNumber += 1

                else:
                    for aValue in aValueList :
                        aList = [ aProperty, '', aValue , aAttribute]
                        aList = map( str, aList )
                        self.theList.append( aList )

            num += 1

    def selectProperty(self, obj, data1, data2, data3):

        aSelectedItem = self.theList[data1]
        aFullPN = None

        print aSelectedItem
        try:
            aFullPropertyName = getFullPN( aSelectedItem[2] )
        except ValueError:
            pass

        if not aFullPropertyName:
            try:
                aFullID = getFullID( aSelectedItem[2] )
                aFullPN = convertFullIDToFullPN( aFullID )
            except ValueError:
                pass
            
        if not aFullPN:
            aFullPN = [ self.theType, self.thePath,
                          self.theID, aSelectedItem[0] ]

        self.theSelected = aFullPN

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
                         ),
             'PropertyAttributes' : ('1','2','3','4','5','6','7','8'),
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
        aPluginManager = Plugin.PluginManager()
        aPropertyWindow = PropertyWindow( 'plugins', simulator(), [fpn,] ,aPluginManager)
        aPropertyWindow.addHandler( 'gtk_main_quit', mainQuit )
        aPropertyWindow.update()

        mainLoop()

    main()











