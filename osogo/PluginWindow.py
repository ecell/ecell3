import os

from config import *
from ViewWindow import *
from gtk import *

class PluginWindow(ViewWindow):
    '''
    self.theRawFullPNList : [ FullPN1, FullID2, FullPN3, , , ]
    theFullPNList()       : [ FullPN1, FullPN2, FullPN3, , , ]
    theFullIDList()       : [ FullID1, FullID2, FullID3, , , ]
    theFullPN()           : FullPN1
    theFullID()           : FullID1
    '''

    def __init__( self, dirname, data, pluginmanager, root=None ):

        self.theClassName = self.__class__.__name__
        aGladeFileName = os.path.join( dirname , self.theClassName + ".glade" )
        ViewWindow.__init__( self, aGladeFileName, root )

        self.thePluginManager = pluginmanager
        self.theSession = self.thePluginManager.theSession 
        self.theDriver = self.theSession.theDriver
        self.theRawFullPNList = data
        self.theTitle = pluginmanager.theInterfaceWindow.theTitle
        	
    def initialize( self, root=None ):

        aMenuWindow = Window( 'PluginWindowPopupMenu.glade', root='menu' )
        self.thePopupMenu = aMenuWindow['menu']

        if root != None:
            self.theClassName = root
            self[self.theClassName].connect( 'button_press_event', self.popupMenu )
            aMenuWindow.addHandlers( { 'copy_fullpnlist'  : self.copyFullPNList,
                                       'paste_fullpnlist' : self.pasteFullPNList,
                                       'add_fullpnlist'   : self.addFullPNList
                                       } )
#        else root !='top_vbox':
        else:
            self.getWidget( self.theClassName )['title'] = self.theTitle

            
            
    def popupMenu( self, widget, aEvent ):

        if aEvent.button == 3:
            self.thePopupMenu.popup( None, None, None, 1, 0 )


    def theFullPNList( self ):

        return map( self.supplementFullPN, self.theRawFullPNList )


    def theFullIDList( self ):

        return map( convertFullPNToFullID, self.theRawFullPNList )


    def theFullPN( self ):

        return self.supplementFullPN( self.theRawFullPNList[0] )


    def theFullID( self ):

        return convertFullPNToFullID( self.theFullPN() )


    def supplementFullPN( self, aFullPN ):

        if aFullPN[PROPERTY] != '' :
            return aFullPN
        else :
            if aFullPN[TYPE] == SUBSTANCE :
                aPropertyName = 'Quantity'
            elif aFullPN[TYPE] == REACTOR :
                aPropertyName = 'Activity'
            elif aFullPN[TYPE] == SYSTEM :
                aPropertyName = 'Activity'
            aNewFullPN = convertFullIDToFullPN( convertFullPNToFullID(aFullPN),
                                                aPropertyName )
            return aNewFullPN


    def theAttributeMap( self ):

        aMap = {}
        for aFullPN in self.theRawFullPNList:
            aFullID = convertFullPNToFullID( aFullPN )
            aPropertyName = aFullPN[PROPERTY]
            aPropertyListFullPN = convertFullIDToFullPN( aFullID, 'PropertyList' )
            aPropertyList = self.theDriver.getProperty( aPropertyListFullPN )
            aAttributeListFullPN = convertFullIDToFullPN( aFullID, 'PropertyAttributes')
            aAttributeList = self.theDriver.getProperty( aAttributeListFullPN )
            num = 0
            for aProperty in aPropertyList:
                aPropertyFullPN = convertFullIDToFullPN( aFullID, aProperty )
                aMap[ aPropertyFullPN ] = aAttributeList[ num ]
                num += 1
        return aMap

        
    def getAttribute( self, aFullPN ):

        aMap = self.theAttributeMap()
        if aMap.has_key( aFullPN ):
            return aMap[ aFullPN ]
        else:
            return 99


    def getValue( self, fullpn ):

        aValueList = self.theDriver.getProperty( fullpn )
        return aValueList[0]


    def setValue( self, fullpn, value ):

        if self.getAttribute( fullpn ) == 3:
            aValueList = ( value, )
            self.theDriver.setProperty( fullpn, aValueList )
            self.thePluginManager.updateAllPluginWindow()
        else:
            aFullPNString = createFullPNString( fullpn )
            self.theSession.printMessage('%s is not settable\n' % aFullPNString )





