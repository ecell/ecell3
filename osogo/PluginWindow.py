import os

from config import *
from ViewWindow import *


class PluginWindow(ViewWindow):
    '''
    theFullPNList()
    theFullIDList()
    theFullPN()
    theFullID()
    '''

    def __init__( self, dirname, sim, data, pluginmanager ):
        aGladeFileName = os.path.join( dirname ,
                                       self.__class__.__name__ + ".glade" )
        ViewWindow.__init__( self, aGladeFileName, sim, data )

        self.thePluginManager = pluginmanager

        self.theRawFullPNList = data

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

