

import ecell.ecs


#FIXME: incomplete
class RecordingSimulator( ecell.ecs.Simulator ):

    def __init__( self, filename ):

        Simulator.__init__( self )
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
        self.record( 'aSimulator.createEntity( \'%s\', %s, \'%s\' )' % (type, fullid, name) )

    def setEntityProperty( self, fullpn, value ):

        self.theSimulator.setEntityProperty(fullpn, value)
        self.record( 'aSimulator.setEntityProperty( %s, %s )' % (fullpn, value) )
        
    def getLogger( self, fullpn ):

        return self.theSimulator.getLogger( fullpn )
        self.record( 'aSimulator.setLogger( %s )' % fullpn )

    def setPendingEventChecker( self, event ):

        self.theSimulator.setPendingEventChecker( event )
        
    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler( event )


if __name__ == "__main__":

    aSimulator = Simulator()









