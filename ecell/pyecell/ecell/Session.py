#! /usr/bin/env python

import Driver


class Session:

    def __init__( self, aDriver ):

        self.thePrintMethod = print
        self.theDriver = aDriver

    def run( self , time='' ):
        self.theDriver.run( time )

    def stop( self ):
        self.theDriver.stop()

    def step( self, num='' ):
        self.theDriver.step( num )

    def getLoggerList( self ):
        return self.theDriver.getLoggerList()

    def getLogger( self, fullpn ):
        return self.theDriver.getLogger( fullpn )

    def setPrintMethod( self, aMethod ):
        self.thePrintMethod = aMethod

    def printMessage( self, message ):
        print message

    def loadModel( self, aModel ):
        pass
        
    def saveModel( self ):
        pass




#FIXME: below are obsolete. should be removed.

class SingleSession( Session ):

    def __init__(self, srfilename ):

        Session.__init__( self, srfilename )
        self.theCondition = 0

    def evaluate(self):

        pass


class OsogoSession( SingleSession ):

    def __init__(self, aMessageWindow, srfilename):

        self.theDriver = Driver.OsogoDriver( srfilename )
        self.theModelInterpreter = ModelInterpreter.ModelInterpreter( self.theDriver )
        self.theMessageWindow = aMessageWindow
        self.theRunningFlag = 0


    def printMessage( self, aMessageString ):

        self.theMessageWindow.printMessage( aMessageString )




class LoopSession( Session ):

    def __init__(self):

        Session.__init__( self )
        self.condition = 0

    def repeat(self):

        while(self.condition):

            self.run

    def run(self):

        pass

    def evaluate(self):

        pass


class TuningSession( LoopSession ):

    def modifyRule( self ):

        pass


class ParallelTuningSession( TuningSession ):

    def parallelize( self ):

        pass





