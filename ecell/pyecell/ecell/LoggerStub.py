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
#
#'Design: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import warnings
from ecell.ObjectStub import ObjectStub

"""
- provides an object-oriented appearance to the ecs.Simulator's Logger API
- does not check validation of each argument.
"""
# FIXME: Logger isn't a PropertyInterface
class LoggerStub( ObjectStub ):
    def __init__( self, aSimulator, aFullPNString ):
        """
        aSimulator : a reference to a Simulator
        aFullPNString : a FullID of the Entity as a String.
       
        This method can throw exceptions.
        """
        ObjectStub.__init__( self, aSimulator )
        self.theLoggingPolicy = None
        
        self.theFullPNString = aFullPNString

    def getName( self ):
        return self.theFullPNString

    def create( self ):
        """
        Actually creates a logger instance.
        """
        if self.exists():
            return
        if self.theLoggingPolicy != None:
            self.theSimulator.createLogger( self.theFullPNString, self.theLoggingPolicy )
        else:
            self.theSimulator.createLogger( self.theFullPNString )

    def delete( self ):
        """
        Deletes the logger created by create()
        """
        self.theSimulator.deleteLogger( self.theFullPNString )

    def exists( self ):
        """
        Checks if the logger is created. Returns true if created,
        false otherwise.
        """
        # When the FullPN of this instance exists in 
        # the FullPN list, returns TRUE
        if self.theFullPNString in self.theSimulator.getLoggerList():
            return True
        else:
            return False

    def getData( self, aStartTime=None, anEndTime=None, anInterval=None ):
        """
        Retrieves the logged data from the associated logger.
        """
        if aStartTime == None:
            aStartTime = self.getStartTime()

        if anEndTime == None:
            anEndTime = self.getEndTime()
        if anInterval == None:
            return self.theSimulator.getLoggerData( self.theFullPNString,
                                                    aStartTime,
                                                    anEndTime )
        else:
            return self.theSimulator.getLoggerData( self.theFullPNString,
                                                    aStartTime,
                                                    anEndTime,
                                                    anInterval )
                                                    
    def getDataWithStartEnd( self, aStartTime, anEndTime ):
        warnings.warn( "Use getData() instead", DeprecationWarning )
        return self.getData( aStartTime, anEndTime )

    def getDataWithStartEndInterval( self, aStartTime, anEndTime, anInterval ):
        warnings.warn( "Use getData() instead", DeprecationWarning )
        return self.getData( aStartTime, anEndTime, anInterval )

    def getStartTime( self ):
        return self.theSimulator.getLoggerStartTime( self.theFullPNString )

    def getEndTime( self ):
        return self.theSimulator.getLoggerEndTime( self.theFullPNString )

    def getSize( self ):
        return self.theSimulator.getLoggerSize( self.theFullPNString )

    def setMinimumInterval( self, anInterval ):
        warnings.warn( "Use setLoggerPolicy instead", DeprecationWarning )
        return self.theSimulator.setLoggerMinimumInterval( self.theFullPNString, anInterval )

    def getMinimumInterval( self ):
        """
        Returns the minimum interval
        This method can throw exceptions.
        """
        warnings.warn( "Use getLoggerPolicy() instead", DeprecationWarning )
        return self.theSimulator.getLoggerMinimumInterval( self.theFullPNString )

    def getLoggerPolicy( self ):
        """
        Returns the logger policy
        This method can throw exceptions.
        """
        return self.theSimulator.getLoggerPolicy( self.theFullPNString )

    def setLoggerPolicy( self, aLoggingPolicy ):
        """
        Takes a tuple of 4 numbers.
          first number: minimum step count.
          second number: minimum time interval.
          third number: policy when disk space or allocataed storage is used up : 0 throw exeption, 1 overwrite old data.
          fourth number: max allocated space by logger in kilobytes.
        Returns the logger policy.
        This method can throw exceptions.
        """
        if self.exists():
            self.theSimulator.setLoggerPolicy( self.theFullPNString, aLoggingPolicy)
        else:
            self.theLoggingPolicy = aLoggingPolicy

