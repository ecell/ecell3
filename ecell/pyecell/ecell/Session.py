#! /usr/bin/env python

import string
import eml
import sys        

from Numeric import *

from os import *
from ecell.ecssupport import *
from ecell.DataFileManager import *
from ecell.ECDDataFile import *

class Session:

    def __init__( self, aSimulator ):

        self.thePrintMethod = self.__plainPrintMethod
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

    def createLogger( self,fullpn ):
        self.theSimulator.getLogger( fullpn )

    def getLogger( self, fullpn ):
        return self.theSimulator.getLogger( fullpn )

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def saveLoggerData( self, aFullPNString='', aStartTime=-1, aEndTime=-1, aInterval=-1, aSaveDirectory='./'):

        # creates instance datafilemanager
        aDataFileManager = DataFileManager()

        # sets root directory to datafilemanager
        aDataFileManager.setRootDirectory(aSaveDirectory)

        aFileIndex=0

        if aFullPNString=='':
            aLoggerNameList = self.getLoggerList()
        else:
            aLoggerNameList = aFullPNString 

        try:#(1)
            for aFullPNString in aLoggerNameList:

                # ---------------------------------------------\----
                # creates filename
                # from [Substance:/CELL/CYTOPLASM:E:Quantity]
                # to   [CYTOPLASM-E-Quantity]
                # ---------------------------------------------\----
                aFileName=split(join(split(aFullPNString,':')[1:],'-'),'/')[-1]
                
                aECDDataFile = ECDDataFile()
                aECDDataFile.setFileName(aFileName)
                aLogger = self.getLogger( aFullPNString )
                
                if aStartTime == -1 or aEndTime == -1:
                    # gets start time and end time from logger
                    aStartTime= aLogger.getStartTime()
                    aEndTime  = aLogger.getEndTime()
                else:
                    # checks the value
                    if not (aLogger.getStartTime() < aStartTime < aLogger.getEndTime()):
                        aStartTime = aLogger.getStartTime()
                    if not (aLogger.getStartTime() < aEndTime < aLogger.getEndTime()):
                        aEndTime = aLogger.getEndTime()

                if aInterval == -1:
                    # gets data with specifing interval
                    aMatrixData = aLogger.getData(aStartTime,aEndTime)
                else:
                    # gets data without specifing interval
                    aMatrixData = aLogger.getData(aStartTime,aEndTime,aInterval)

                # sets data name
                aECDDataFile.setDataName(aFullPNString)

                # sets matrix data
                aECDDataFile.setMatrixData(aMatrixData)

                # -------------------------------------------------
                # adds data file to data file manager
                # -------------------------------------------------
                aDataFileManager.getFileMap()[`aFileIndex`] = aECDDataFile

                aFileIndex = aFileIndex + 1
        
        except:#(1)
            
            import sys
            self.__plainPrintMethod( __name__ )
            self.__plainPrintMethod( sys.exc_traceback )
            aErrorMessage= "Error : could not save [%s] " %aFullPNString
            self.__plainPrintMethod( aErrorMessage )
            return None

        aDataFileManager.saveAll()         


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


    def loadScript( self, ecs ):
        execfile(ecs)


    def __plainPrintMethod( aMessage ):
        print aMessage


    def loadModel( self, aFileObject ):
        self.theEml = eml.Eml( aFileObject )

        self.__loadStepper()
        self.__loadEntity()
        self.__loadStepperProperty()
        self.__loadProperty()

        
    def saveModel( self ):
        pass


    def __loadStepper( self ):
        """stepper loader"""

        aStepperList = self.theEml.getStepperList()

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



    def __loadEntity( self ):

        anEntityList = self.theEml.getEntityList()

        for anEntity in anEntityList:

            self.theSimulator.createEntity( anEntity[ 'Type' ], \
                                            anEntity[ 'FullId' ], \
                                            anEntity[ 'Name' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.createEntity( '" + anEntity[ 'Type' ] + "', '" + \
            #         anEntity[ 'FullId' ] + "', '" + anEntity[ 'Name' ]   + "' )"
            #print script
            ##=============================================================================



    def __loadStepperProperty( self ):
        aStepperPropertyList = self.theEml.getStepperPropertyList()
        
        for aStepperProperty in aStepperPropertyList:

            self.theSimulator.setEntityProperty( str( aStepperProperty[ 'FullPn' ] ), \
                                           aStepperProperty[ 'StepperId' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.setEntityProperty( '" + aStepperProperty[ 'FullPn' ] + "', " +\
            #         str( aStepperProperty[ 'StepperId' ] ) + ')'
            #print script
            ##=============================================================================        
            


    def __loadProperty( self ):
        aPropertyList = self.theEml.getEntityPropertyList()
        
        for aProperty in aPropertyList:

            self.theSimulator.setEntityProperty( aProperty[ 'FullPn' ], \
                                           aProperty[ 'ValueList' ] )

            ## Debug for Output -----------------------------------------------------------
            #script = "self.theSimulator.setEntityProperty( '" +str( aProperty[ 'FullPn' ] ) + "', " + \
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





