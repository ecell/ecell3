#! /usr/bin/python

import Driver
import ModelInterpreter

class Session:

    def __init__( self ):

        self.theDriver = Driver.SimpleDriver()
        self.theSimulator = self.theDriver.theSimulator
        self.theModelInterpreter = ModelInterpreter.ModelInterpreter( self.theDriver )

    def printMessage( self, message ):
        print message

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

class SingleSession( Session ):

    def __init__(self):

        Session.__init__( self )
        self.condition = 0


    def evaluate(self):

        pass


class GuiSession( SingleSession ):

    def __init__(self, aMessageWindow):

        SingleSession.__init__( self )
        self.theMessageWindow = aMessageWindow

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





