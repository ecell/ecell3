#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
from BufferFactory import *
from Utils import *

class Command:

    ARG_NO = 0
    def __init__(self, aReceiver, *args):
        """
        checks and stores arguments     
        after a buffer is passed to a Command, it shouldnt be 
        changed by other classes
        """
        self.theArgs = copyValue( args)
        self.theReceiver = aReceiver
        self.executed = False
        self.theReverseCommandList = None
        self.doMultiplex = True

    def reset( self ):
        self.executed = False
        self.theReverseCommandList = None
    
    def makeExecuted( self, reverseCommandList ):
        #the effects of command has already taken place
        self.checkArgs()
        self.executed = True
        self.reverseCommandList = reverseCommandList


    def execute(self):
        """
        executes stored command
        can only be executed once
        returns True if successful
        returns False if command is non executable
        """
        if self.isExecuted():
            return
        if self.isExecutable() :

            self.createReverseCommand()
            if self.do():
                self.executed = True
            else:
                raise Exception("%s command failed.\n Arguments: %s"%(self.__class__.__name__, self.theArgs) )
                self.theReverseCommandList = None
        else:
            raise Exception("%s command argumentcheck failed. Cannot execute.\n Arguments: %s"%(self.__class__.__name__, self.theArgs) )



    def isExecuted(self):
        return self.executed



    def isExecutable(self):
        return  self.checkArgs()


    def getReverseCommandList(self):
        """
        creates and returns a reverse commandlist with Buffers
        can only be called after execution
        """
        return self.theReverseCommandList


    def checkArgs( self ):
        """
        return True if self.Args are valid for this command
        """

        if len(self.theArgs) != self.ARGS_NO :

            return False
        return True


    def do(self):
        """
        perform command
        return True if successful
        """
        return True

    def createReverseCommand(self):
        """
        create  reverse command instance(s) and store it in a list as follows:
        """
        self.theReverseCommandList = [ Command( self.theReceiver, [] ) ]

    def getAffectedObject( self ):
        if self.executed:
            return self.getAffected()
        else:
            return ( None, None )

    
    def getSecondAffectedObject( self ):
        if self.executed:
            return self.getAffected2()
        else:
            return ( None, None )


    def getAffected( self ):
        return ( None, None )

    def getAffected2( self ):
        return ( None, None )
    
    def doNotMultiplexReverse( self ):
        if type(self.theReverseCommandList) == type([]):
            for aReverseCmd in self.theReverseCommandList:
                if type(aReverseCmd) != type(self):
                    continue
                aReverseCmd.doNotMultiplex()
                
    def doNotMultiplex( self ):
        self.doMultiplex = False
        

class ModelCommand( Command ):

    """
    contains the command name and the buffer needed to execute it
    can execute the command which can be:
    """

    def checkArgs( self ):

        if not Command.checkArgs(self):
            return False

        if type( self.theReceiver) == type(self):

            if self.theReceiver.__class__.__name__ == 'ModelEditor':


                self.theModel = self.theReceiver.getModel()
                self.theBufferFactory = BufferFactory ( self.theModel )
                self.theBufferPaster = BufferPaster ( self.theModel )

                return True

        return False



    def isFullPNExist( self, aFullPN ):

        # first check whether FullID exists
        aFullID = getFullID( aFullPN )
        if not self.theModel.isEntityExist(aFullID ):
            return False
        propertyList = self.theModel.getEntityPropertyList( aFullID )
        if getPropertyName( aFullPN ) not in propertyList:
            return False
        return True

