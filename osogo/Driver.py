#! /usr/bin/python

import ecs
import gtk

from ecssupport import *

class Driver:

    def __init__( self ):

        self.theSimulator = ecs.Simulator()
        self.theSimulator.setPendingEventChecker( gtk.events_pending )
        self.theSimulator.setEventHandler( gtk.mainiteration )

    def initialize( self ):
        self.theSimulator.initialize()

    def createEntity( self, type, fullid, name ):
        self.theSimulator.createEntity( type, fullid, name )

    def getProperty( self, fullpn ):
        return self.theSimulator.getProperty(fullpn)

    def setProperty( self, fullpn, value ):
        self.theSimulator.setProperty(fullpn, value)

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def getLogger( self, fullpn ):
        logger = self.theSimulator.getLogger( fullpn )
#        logger = Logger( fullpn )
        return logger

    def setPendingEventChecker( self, event ):
        self.theSimulator.setPendingEventChecker( event )

    def setEventHandler( self, event ):
        self.theSimulator.setEventHandler(event )

    def printProperty( self, fullpn ):
        value = self.theSimulator.getProperty( fullpn )
        print fullpn, '\t=\t', value

#    def printProperty( self, fullpn ):
#        if fullpn[3] == 'ActivityPerSecond':
#            value = ' ... failed to print'
#        else:
#            value = self.theSimulator.getProperty( fullpn )
#        print fullpn, '\t=\t', value

    def printAllProperties( self, fullid ):
        plistfullpn = convertFullIDToFullPN( fullid, 'PropertyList' )
        properties = self.theSimulator.getProperty( plistfullpn )
        for property in properties:
            fullpn = convertFullIDToFullPN( fullid, property )
            try:
                fullpn = convertFullIDToFullPN( fullid, property )
                self.printProperty( fullpn )
            except:
                print "failed to print %s:%s" % ( fullid, property )

    def printList( self, primitivetype, systempath, list ):
        for i in list:
            self.printAllProperties( ( primitivetype, systempath, i ) )

    def getCurrentTime( self ):
        time = self.theSimulator.getProperty( (SYSTEM, '/', '/', 'CurrentTime' ) )
        return time[0]


class SimpleDriver( Driver ):

    pass

class Logger:

    def __init__( self, fpn ):
        self.theLogger = ecs.getLogger( fpn )

    def getStartTime( self ):
        return self.theLogger.getStartTime()

    def getEndTime( self ):
        return self.theLogger.getEndTime()

    def getData( self, start=0, end=0, interval=0 ):
        return self.theLogger.getData( start, end, interval )

class LoggerTest( Logger ):

    def getStartTime( self ):
        return 0

    def getEndTime( self ):
        return 100

    def getData( self, start=0, end=0, interval=0 ):
        array = ((0, 0), (1, 20), (2, 30), (3, 35), (4, 40), (5, 15), (6, 10), (7, 15))
        return array

if __name__ == "__main__":

    class MainWindow:

        def __init__( self ):
            self.theSimulator = simulator()


    class Simulator:

        def __init__( self ):

            self.dic={('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity') : (1950,),}

        def initialize( self ):
            pass

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


    aDriver = Driver()

    print aDriver.getProperty(('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity'))
    aDriver.setProperty(('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity'), (2000,))
    print aDriver.getProperty(('Substance', '/CELL/CYTOPLASM', 'ATP','Quantity'))

    for aLoggerFpn in  aDriver.getLoggerList():
        aLogger = aDriver.getLogger( aLoggerFpn )
        print aLogger.getStartTime()
        print aLogger.getEndTime()

