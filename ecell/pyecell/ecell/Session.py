#! /usr/bin/env python

import string
import eml
import sys
import os

from Numeric import *
import ecell.ecs

from ecell.ecssupport import *
import ecell.DataFileManager
import ecell.ECDDataFile

#from ecell.FullID import *
#from ecell.util import *
#from ecell.ECS import *

class Session:
    'Session class'


    def __init__( self, aSimulator ):
        'constructor'

        self.theMessageMethod = self.__plainMessageMethod
        self.theSimulator = aSimulator


    #
    # Session methods
    #

    def loadScript( self, ecs, parameters={} ):

        aContext = self.__createScriptContext( parameters )

        execfile( ecs, aContext )
            
    def interact( self, parameters={} ):

        aContext = self.__createScriptContext( parameters )
        
        import readline # to provide convenient commandline editing :)
        import code
        anInterpreter = code.InteractiveConsole( aContext )
        anInterpreter.interact( 'ecell3-session' )


    def loadModel( self, aFile ):
        if type( aFile ) == str:
            aFileObject = open( aFile )
        else:
            aFileObject = aFile

        anEml = eml.Eml( aFileObject )

        self.__loadStepper( anEml )

        # load root system properties
        aPropertyList = anEml.getEntityPropertyList( 'System::/' )
        self.__loadEntityPropertyList( anEml, 'System::/', aPropertyList )

        self.__loadEntity( anEml )

        self.theSimulator.initialize()

        
    def saveModel( self ):
        pass

    def setMessageMethod( self, aMethod ):
        self.theMessageMethod = aMethod

    def message( self, message ):
        self.theMessageMethod( message )

    def setLogMethod( self, aMethod ):
        print 'setLogMethod() is deprecated. use setMessageMethod instead.'
        self.theMessageMethod = aMethod

    def log( self, message ):
        print 'log() is deprecated. use message() instead'
        self.theMessageMethod( message )



    #
    # Simulator methods
    #
    
    def run( self , time='' ):
        if not time:
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )

    def stop( self ):
        self.theSimulator.stop()

    def step( self, num=None ):
        if not num:
            self.theSimulator.step()
        else:
            # Simulator should support step( num )
            for i in xrange(num):
                self.theSimulator.step()

    def getCurrentTime( self ):
        return self.theSimulator.getCurrentTime()

    def setEventChecker( self, event ):
        self.theSimulator.setEventChecker( event )

    def setEventHandler( self, event ):
        self.theSimulator.setEventHandler( event )

    # no need to initialize explicitly in current version
    def initialize( self ):
        self.theSimulator.initialize()


    #
    # Stepper methods
    #


    def getStepperList():
        return self.theSimulator.getStepperList()

    def createStepperStub( self, id ):
        return StepperStub( self.theSimulator, id )


    #
    # Entity methods
    #

    def getEntityList( self, entityType, systemPath ):
        return self.theSimulator.getEntityList( entityType, systemPath )

    def createEntityStub( self, fullid ):
        return EntityStub( self.theSimulator, fullid )


    #
    # Logger methods
    #

    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()
        
    def createLoggerStub( self, fullpn ):
        return LoggerStub( self.theSimulator, fullpn )

    def saveLoggerData( self, aFullPNString='', aStartTime=-1, aEndTime=-1, aInterval=-1, aSaveDirectory='./Data'):

        try:
            os.mkdir( aSaveDirectory )
        # creates instance datafilemanager
        except:
            self.log( "\'" + aSaveDirectory + "\'" + " file exists." )
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
                # from [Variable:/CELL/CYTOPLASM:E:Value]
                # to   [CYTOPLASM-E-Value]
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
            ## self.message( __name__ )
            self.message( sys.exc_traceback )
            aErrorMessage= "Error : could not save [%s] " %aFullPNString
            self.message( aErrorMessage )
            return None

        aDataFileManager.saveAll()         
        self.message( "All files are saved." )



    #
    # private methods
    #

    def __plainMessageMethod( self, aMessage ):
        print aMessage


    def __loadStepper( self, anEml ):
        """stepper loader"""

        aStepperList = anEml.getStepperList()

        for aStepper in aStepperList:

            aClassName = anEml.getStepperClass( aStepper )
            self.theSimulator.createStepper( str( aClassName ),\
                                             str( aStepper ) )

            aPropertyList = anEml.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                
                aValue = anEml.getStepperProperty( aProperty )
                self.theSimulator.setStepperProperty( aStepper,\
                                                      aProperty,\
                                                      aValue )
                                             
    def __loadEntity( self, anEml, aSystemPath='/' ):

        aVariableList = anEml.getEntityList( 'Variable', aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System',    aSystemPath )

        self.__loadEntityList( 'Variable', aSystemPath, aVariableList )
        self.__loadEntityList( 'Process',   aSystemPath, aProcessList )
        self.__loadEntityList( 'System',    aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadEntity( aSubSystemPath )


    def __loadEntityList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):
        
        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = anEml.getEntityClass( aFullID )
            self.theSimulator.createEntity( str( aClassName ), aFullID )

            aPropertyList = anEml.getEntityPropertyList( aFullID )

            self.__loadEntityPropertyList( aFullID, aPropertyList )


    def __loadEntityPropertyList( self, anEml, aFullID, aPropertyList ):

        for aProperty in aPropertyList:
            aFullPN = aFullID + ':' + aProperty
            aValue = anEml.getEntityProperty( aFullPN )
            self.theSimulator.setEntityProperty( aFullPN, aValue )

    def __createScriptContext( self, parameters ):

        # theSession == self in the script
        aContext = { 'theSession': self }
        
        # flatten class methods and object properties so that
        # 'self.' isn't needed for each method calls in the script
        aDict = {}
        for aKey in self.__dict__.keys() + self.__class__.__dict__.keys():
            aDict[ aKey ] = getattr( self, aKey )

        aContext.update( aDict )
            
        # add parameters to the context
        aContext.update( parameters )

        return aContext


if __name__ == "__main__":
    pass
