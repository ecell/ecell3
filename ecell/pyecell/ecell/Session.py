#! /usr/bin/env python
import string
import eml
import os
import sys        

class Session:

    def __init__( self, aSimulator ):

        self.thePrintMethod = self.plainPrintMethod
        self.theSimulator = aSimulator

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

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def getLogger( self, fullpn ):
        return self.theSimulator.getLogger( fullpn )

    def setPendingEventChecker( self, event ):

        self.theSimulator.setPendingEventChecker( event )

    def setEventHandler( self, event ):

        self.theSimulator.setEventHandler( event )

    def getCurrentTime( self ):

        return self.theSimulator.getCurrentTime()

    def setPrintMethod( self, aMethod ):
        self.thePrintMethod = aMethod

    def printMessage( self, message ):
        self.thePrintMethod( message )


    def loadModel( self, aFileObject ):
        self.__theEml = eml.Eml( aFileObject )

        self.loadStepper()
        self.loadEntity()
        self.loadStepperProperty()
        self.loadProperty()

        
    def saveModel( self ):
        pass

    def loadScript( self, ecs ):
        execfile(ecs)


    def plainPrintMethod( aMessage ):
        print aMessage




    def loadStepper( self ):
        """stepper loader"""

        aStepperList = self.__theEml.getStepperList()

        for aStepper in aStepperList:

            self.theSimulator.createStepper( aStepper[ 'Class' ], \
                                             aStepper[ 'Id' ] )

            ## ERROR: don't use the third value ... why? (020719)
            #self.theSimulator.createStepper( aStepper[ 'Class' ], \
            #                                 aStepper[ 'Id' ], \
            #                                 aStepper[ 'ValueList' ] )

            ## Debug for Output --------------------------------------
            #script = "self.theSimulator.createStepper( '" +\
            #         aStepper[ 'Class' ] + "', '" +\
            #         aStepper[ 'Id' ] + "', " +\
            #         str( aStepper[ 'ValueList' ] ) + " )"
            #print script
            ## ========================================================



    def loadEntity( self ):

        anEntityList = self.__theEml.getEntityList()

        for anEntity in anEntityList:

            self.theSimulator.createEntity( anEntity[ 'Type' ], \
                                            anEntity[ 'FullId' ], \
                                            anEntity[ 'Name' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.createEntity( '" + anEntity[ 'Type' ] + "', '" + \
            #         anEntity[ 'FullId' ] + "', '" + anEntity[ 'Name' ]   + "' )"
            #print script
            ##=============================================================================



    def loadStepperProperty( self ):
        aStepperPropertyList = self.__theEml.getStepperPropertyList()
        
        for aStepperProperty in aStepperPropertyList:

            print 'DEBUG:', aStepperProperty ##DebugMessage

            self.theSimulator.setProperty( str( aStepperProperty[ 'FullPn' ] ), \
                                           aStepperProperty[ 'StepperId' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.setProperty( '" + aStepperProperty[ 'FullPn' ] + "', " +\
            #         str( aStepperProperty[ 'StepperId' ] ) + ')'
            #print script
            ##=============================================================================        
            


    def loadProperty( self ):
        aPropertyList = self.__theEml.getPropertyList()
        
        for aProperty in aPropertyList:

            self.theSimulator.setProperty( aProperty[ 'FullPn' ], \
                                           aProperty[ 'ValueList' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.setProperty( '" +str( aProperty[ 'FullPn' ] ) + "', " + \
            #         str( aProperty[ 'ValueList' ] ) + ')'
            #
            #print script
            ##=============================================================================





if __name__ == "__main__":

    import ecs

    aSession = Session( ecs.Simulator() )
    aSimulator = aSession.theSimulator
    anEmlFileName = sys.argv[1]
    anEcsFileName = sys.argv[2]

    aFile = open( anEmlFileName )
    
    aSession.loadModel( aFile )
    aFile.close()
    
    aSession.loadScript( anEcsFileName )





