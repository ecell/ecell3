#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER


import ecell.emc


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

    def setEventChecker( self, event ):

        self.theSimulator.setEventChecker( event )
        
    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler( event )


if __name__ == "__main__":

    aSimulator = Simulator()









