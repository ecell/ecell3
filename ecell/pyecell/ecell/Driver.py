#! /usr/bin/python

import ecs
import gtk
import types

from ecssupport import *

class Driver:

    def __init__( self ):

        self.theSimulator = ecs.Simulator()
        self.theSimulator.setPendingEventChecker( gtk.events_pending )
        self.theSimulator.setEventHandler( gtk.mainiteration )
        self.initialize()

    def run( self , time='' ):

        if not time:
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )

    def stop( self ):

        self.theSimulator.stop()

    def step( self, num='' ):

        if not num:
            self.theSimulator.step()
        else:
            for i in range(num):
                self.theSimulator.step()

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

        return self.theSimulator.getLogger( fullpn )

    def setPendingEventChecker( self, event ):

        self.theSimulator.setPendingEventChecker( event )

    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler(event )

    def printProperty( self, fullpn ):

        value = self.theSimulator.getProperty( fullpn )
        print fullpn, '\t=\t', value

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

    def isNumber( self, aFullPN ):

        aValue = self.getProperty( aFullPN )
        if type( aValue[0] ) is types.IntType:
            return 1
        elif type( aValue[0] ) is types.FloatType:
            return 1
        else:
            return 0


class SessionRecorder:

    def __init__( self, filename ):

        self.theOutput = open(filename, 'w')
        
    def record( self, string ):

        self.theOutput.write( string )
        self.theOutput.write( "\n" )

    def __dell__( self ):

        self.theOutput.close()
    


class StandardDriver( Driver, SessionRecorder ):

    def __init__( self, srfilename ):

        SessionRecorder.__init__( self, srfilename )
        Driver.__init__( self )

    def run( self , time='' ):

        if not time:
            self.theStartTime = self.getCurrentTime()
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )
            self.record( 'aSession.run( %f )' % time )

    def stop( self ):

        self.theSimulator.stop()
        aRunTime = self.getCurrentTime() - self.theStartTime
        self.record( 'aSession.run( %f )' % aRunTime )

    def step( self, num = 1 ):

        for i in range(num):
            self.theSimulator.step()
        self.record( 'for i in range( %d ):' % num )
        self.record( '    aSession.step()' )

    def initialize( self ):

        self.theSimulator.initialize()

    def createEntity( self, type, fullid, name ):

        self.theSimulator.createEntity( type, fullid, name )
        self.record( 'aDriver.createEntity( \'%s\', %s, \'%s\' )' % (type, fullid, name) )

    def setProperty( self, fullpn, value ):

        self.theSimulator.setProperty(fullpn, value)
        self.record( 'aDriver.setProperty( %s, %s )' % (fullpn, value) )
        
    def getLogger( self, fullpn ):

        return self.theSimulator.getLogger( fullpn )
        self.record( 'aDriver.setLogger( %s )' % fullpn )

    def setPendingEventChecker( self, event ):

        self.theSimulator.setPendingEventChecker( event )
        
    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler( event )


class OsogoDriver( StandardDriver ):

    def __init__( self, srfilename ):

        StandardDriver.__init__( self, srfilename )
        self.record('aSession = aMainWindow.theSession')
        self.record('aDriver = aSession.theDriver')
        self.record('aPluginManager = aMainWindow.thePluginManager')
        self.record('#--------------------------------------------------')


if __name__ == "__main__":

    aDriver = Driver()









