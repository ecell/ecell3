#!/usr/bin/env python

from Window import *
from gtk import *

from ecssupport import *

class LoggerWindow(Window):

    def __init__( self, aMainWindow ):

        Window.__init__( self, 'LoggerWindow.glade' )

        self.theMainWindow = aMainWindow
        self.theFullPNList = self.theMainWindow.theSimulator.getLoggerList()

        self.theEntryList = self.getWidget( 'clist3' )
        self.theList = []
        self.initialize()

    def initialize( self ):
        self.setEntryList()

    def setEntryList( self ):
        for fpn in self.theFullPNList :
            theFullPN = createFullPNString( fpn )
            aLogger = self.theMainWindow.theSimulator.getLogger( fpn )
            start = aLogger.getStartTime()
            end = aLogger.getEndTime()
            aList = [ theFullPN, start, end ]
            aList = map( str, aList)
            self.theList.append( aList )

        self.update()

    def update( self ):
        self.theEntryList.clear()
        for aValue in self.theList:
            self.theEntryList.append( aValue )
            

### test code

if __name__ == "__main__":

    class MainWindow:

        def __init__( self ):
            self.theSimulator = simulator()

    class simulator:

        def __init__( self ):

            self.dic={('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

        def getProperty( self, fpn ):
            return self.dic[fpn]

        def setProperty( self, fpn, value ):
            self.dic[fpn] = value

        def getLogger( self, fpn ):
            logger = Logger( fpn )
            return logger

        def getLoggerList( self ):
            fpnlist = ((SUBSTANCE, '/CELL/CYTOPLASM', 'ATP', 'Quantity'),
                       (SUBSTANCE, '/CELL/CYTOPLASM', 'ADP', 'Quantity'),
                       (SUBSTANCE, '/CELL/CYTOPLASM', 'AMP', 'Quantity'))
            return fpnlist

    class Logger:

        def __init__( self, fpn ):
            pass

        def getStartTime( self ):
            return 0

        def getEndTime( self ):
            return 100

               
    def mainQuit( obj, data ):
        gtk.mainquit()
        
    def mainLoop():
        # FIXME: should be a custom function

        gtk.mainloop()

    def main():
        aMainWindow = MainWindow()
        aLoggerWindow = LoggerWindow( aMainWindow )
        mainLoop()

    main()





