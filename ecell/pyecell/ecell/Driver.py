#! /usr/bin/python

import ecs
import types

from ecssupport import *

class Driver:

    def __init__( self ):

        self.theSimulator = ecs.Simulator()

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

    def getCurrentTime( self ):

        time = self.theSimulator.getProperty( (SYSTEM, '/', '/', 'CurrentTime' ) )
        return time[0]


# FIXME:
# below are util functions. should be moved to other file

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

    def isNumber( self, aFullPN ):

        aValue = self.getProperty( aFullPN )
        if type( aValue[0] ) is types.IntType:
            return 1
        elif type( aValue[0] ) is types.FloatType:
            return 1
        else:
            return 0




#FIXME: incomplete
class RecordingDriver( Driver ):

    def __init__( self, filename ):

        Driver.__init__( self )
        self.theOutput = open(filename, 'w')

    def __del__( self ):

        self.theOutput.close()

        
    def record( self, string ):

        self.theOutput.write( string )
        self.theOutput.write( "\n" )


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


if __name__ == "__main__":

    aDriver = Driver()









