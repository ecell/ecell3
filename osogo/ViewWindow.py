#!/usr/bin/env python

from Window import *

import string
import sys
from ecssupport import *

class ViewWindow(Window):
    '''
    theFullPNList()
    theFullIDList()
    theFullPN()
    theFullID()
    '''

    theClipBoard = ''

    def __init__(self, gladefile, sim, fpns):
        Window.__init__( self, gladefile )

        self.theSimulator = sim
        
        self.theRawFullPNList = fpns
        
        self.addHandlers(
            { 'copy_fqpps':            self.copyFPNs,
              'paste_fqpps':           self.pasteFPNs,
              'add_fqpps':             self.addFPNs,
              'print_fqpps_to_stdout': self.printFPNs
              } )

    def update( self ):
        pass

    def initialize( self ):
        pass

    def printTest( self, obj, data ):
        print obj, data

    def copyFPNs(self, a, b):
        ViewWindow.theClipBoard = self.theRawFullPNList
        print 'copy'

    def pasteFPNs(self, a, b):
        self.theRawFullPNList = ViewWindow.theClipBoard
        initialize()
        print 'paste' + ' : ' + self.theFullRawPNList

    # overwrite in subclass if needed
    def addFPNs(self, a, b):
        self.theRawFullPNList.extend( ViewWindow.theClipBoard )
        print 'add' + ' : ' + self.theRawFullPNList

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

    def printFPNs(self, a, b):
        print self.theFullPNList

if __name__ == "__main__":
    def mainLoop():
        gtk.mainloop()

    def main():
        mainLoop()

    main()

















