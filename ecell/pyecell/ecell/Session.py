#! /usr/bin/env python

import string
import eml
import sys
import os

from Numeric import *
import ecell.Session
import ecell.ecs

from ecell.ecssupport import *
import ecell.DataFileManager
import ecell.ECDDataFile

from ecell.FullID import *
from ecell.util import *
from ecell.ECS import *

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

    def initialize( self ):
        self.theSimulator.initialize()

    def createLogger( self,fullpn ):
        #self.theSimulator.getLogger( fullpn )

	for aLogger in self.theSimulator.getLoggerList():
		if aLogger == fullpn:
			return None

        self.theSimulator.createLogger( fullpn )

    def getLogger( self, fullpn ):
        return self.theSimulator.getLogger( fullpn )

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def saveLoggerData( self, aFullPNString='', aStartTime=-1, aEndTime=-1, aInterval=-1, aSaveDirectory='./Data'):

        try:
            os.mkdir( aSaveDirectory )
        # creates instance datafilemanager
        except:
            print "\'" + aSaveDirectory + "\'" + " file exists."
        aDataFileManager = ecell.DataFileManager.DataFileManager()

        # sets root directory to datafilemanager
        aDataFileManager.setRootDirectory(aSaveDirectory)

        aFileIndex = 0

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
                aFileName=string.split(string.join(string.split(aFullPNString,':')[1:],'-'),'/')[-1]
                
                aECDDataFile = ecell.ECDDataFile.ECDDataFile()
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
            ## self.__plainPrintMethod( __name__ )
            self.__plainPrintMethod( sys.exc_traceback )
            aErrorMessage= "Error : could not save [%s] " %aFullPNString
            self.__plainPrintMethod( aErrorMessage )
            return None

        aDataFileManager.saveAll()         
        print "All files are saved."

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

        # load root system properties
        aPropertyList = self.theEml.getEntityPropertyList( 'System::/' )
        self.__loadEntityPropertyList( 'System::/', aPropertyList )

        self.__loadEntity()

        
    def saveModel( self ):
        pass


    def __loadStepper( self ):
        """stepper loader"""

        aStepperList = self.theEml.getStepperList()

        for aStepper in aStepperList:

            aClassName = self.theEml.getStepperClass( aStepper )
            self.theSimulator.createStepper( str( aClassName ),\
                                             str( aStepper ) )

            aPropertyList = self.theEml.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                
                aValue = self.theEml.getStepperProperty( aProperty )
                self.theSimulator.setStepperProperty( aStepper,\
                                                      aProperty,\
                                                      aValue )
                                             
    def __loadEntity( self, aSystemPath='/' ):

        aSubstanceList = self.theEml.getEntityList( 'Substance', aSystemPath )
        aReactorList   = self.theEml.getEntityList( 'Reactor',   aSystemPath )
        aSubSystemList = self.theEml.getEntityList( 'System',    aSystemPath )

        self.__loadEntityList( 'Substance', aSystemPath, aSubstanceList )
        self.__loadEntityList( 'Reactor',   aSystemPath, aReactorList )
        self.__loadEntityList( 'System',    aSystemPath, aSubSystemList )

        print aSubSystemList
        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadEntity( aSubSystemPath )


    def __loadEntityList( self, anEntityTypeString, aSystemPath, anIDList ):
        
        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = self.theEml.getEntityClass( aFullID )
            self.theSimulator.createEntity( str( aClassName ), aFullID )

            aPropertyList = self.theEml.getEntityPropertyList( aFullID )

            self.__loadEntityPropertyList( aFullID, aPropertyList )



    def __loadEntityPropertyList( self, aFullID, aPropertyList ):

        for aProperty in aPropertyList:
            aFullPN = aFullID + ':' + aProperty
            aValue = self.theEml.getEntityProperty( aFullPN )
            self.theSimulator.setEntityProperty( aFullPN, aValue )



if __name__ == "__main__":
    pass
