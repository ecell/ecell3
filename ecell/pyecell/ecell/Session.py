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

BANNERSTRING =\
'''ecell3-session [ E-Cell SE Version %s, on Python Version %d.%d.%d ]
Copyright (C) 1996-2003 Keio University.
Written by Kouichi Takahashi <shafi@e-cell.org>'''\
% ( ecell.ecs.getLibECSVersion(), sys.version_info[0], sys.version_info[1], sys.version_info[2] )


class Session:
    'Session class'


    def __init__( self, aSimulator ):
        'constructor'

        self.theMessageMethod = self.__plainMessageMethod
        self.theSimulator = aSimulator

        self.theModelName = ''


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

        self._prompt = self._session_prompt( self )
        anInterpreter.runsource( 'import sys; sys.ps1=theSession._prompt; del sys' )

        anInterpreter.interact( BANNERSTRING )


    def loadModel( self, aModel ):
        # aModel : an EML instance, a file name (string) or a file object
        # return -> None
        # This method can thwor exceptions. 

        # checks the type of aModel

        # if the type is EML instance
        if type( aModel ) == type( eml.Eml ):
            anEml = aModel
            aModelName = '<eml.Eml>'  # what should this be?

        # if the type is string
        elif type( aModel ) == str:
            aFileObject = open( aModel )
            aModelName = aModel
            anEml = eml.Eml( aFileObject )

        # if the type is file object
        elif type( aModel ) == file:
            aFileObject = aModel
            aModelName = aModel.name
            anEml = eml.Eml( aFileObject )

        # When the type doesn't match
        else:
            raise TypeError, " The type of aModel must be EML instance, string(file name) or file object "

        # calls load methods
        self.__loadStepper( anEml )
        self.__loadEntity( anEml )
        self.__loadProperty( anEml )

        # saves ModelName 
        self.theModelName = aModelName

        # initializes Simulator
        self.theSimulator.initialize()

    # end of loadModel
        

    def saveModel( self , aModel ):
        # aModel : a file name (string) or a file object
        # return -> None
        # This method can thwor exceptions. 
        
        # creates ana seve an EML instance 
        self.anEml = eml.Eml()

        # creates root entity
        self.anEml.createEntity('System', 'System::/')
        
        # calls save methods
        self.__saveEntity()
        self.__saveStepper()
        self.__saveProperty()

        # if the type is string
        if type( aModel ) == str:
       	    aString = self.anEml.asString()
            aFileObject = open( aModel, 'w' )
            aFileObject.write( aString )
            aFileObject.close()

        # if the type is file object
        elif type( aModel ) == file:
       	    aString = self.anEml.asString()
            aFileObject = aModel
            aFileObject.write( aString )
            aFileObject.close()

        # When the type doesn't match
        else:
            raise TypeError, " The type of aModel must be string(file name) or file object "

    # end of saveModel


        
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

    def step( self, num=1 ):
        self.theSimulator.step( num )

    def getNextEvent( self ):
        return self.theSimulator.getNextEvent()

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


    def getStepperList( self ):
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
        
    def createLogger( self, fullpn ):
        self.message( 'createLogger method will be deprecated. Use LoggerStub.' )
        aStub = self.createLoggerStub( fullpn )
        aStub.create()

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
                
                aValue = anEml.getStepperProperty( aStepper, aProperty )
                self.theSimulator.setStepperProperty( aStepper,\
                                                      aProperty,\
                                                      aValue )
                                             
    def __loadEntity( self, anEml, aSystemPath='/' ):

        aVariableList = anEml.getEntityList( 'Variable', aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System',    aSystemPath )

        self.__loadEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__loadEntityList( anEml, 'Process',  aSystemPath, aProcessList )
        self.__loadEntityList( anEml, 'System',   aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadEntity( anEml, aSubSystemPath )


    def __loadProperty( self, anEml, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = anEml.getEntityList( 'Variable',  aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process',   aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System',    aSystemPath )

        self.__loadPropertyList( anEml, 'Variable',\
                                 aSystemPath, aVariableList )
        self.__loadPropertyList( anEml, 'Process',  aSystemPath, aProcessList )
        self.__loadPropertyList( anEml, 'System',\
                                 aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__loadProperty( anEml, aSubSystemPath )

    def __loadPropertyList( self, anEml, anEntityTypeString,\
                            aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID

            aPropertyList = anEml.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                aValue = anEml.getEntityProperty( aFullPN )
                self.theSimulator.setEntityProperty( aFullPN, aValue )

        

    def __loadEntityList( self, anEml, anEntityTypeString,\
                          aSystemPath, anIDList ):
        
        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = anEml.getEntityClass( aFullID )
            self.theSimulator.createEntity( str( aClassName ), aFullID )



    def __createScriptContext( self, parameters ):

        # theSession == self in the script
        aContext = { 'theSession': self, 'self': self }
        
        # flatten class methods and object properties so that
        # 'self.' isn't needed for each method calls in the script
        aKeyList = list ( self.__dict__.keys() +\
                          self.__class__.__dict__.keys() )

        aDict = {}
        for aKey in aKeyList:
            aDict[ aKey ] = getattr( self, aKey )

        aContext.update( aDict )
            
        # add parameters to the context
        aContext.update( parameters )

        return aContext

    def __saveStepper( self ):
        """stepper loader"""

        aStepperList = self.theSimulator.getStepperList()

        for aStepper in aStepperList:

            aClassName = self.theSimulator.getStepperClassName( aStepper )
            self.anEml.createStepper( str( aClassName ),\
                                             str( aStepper ) )

            aPropertyList = self.theSimulator.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                
                aValue = self.theSimulator.getStepperProperty( aStepper, aProperty )
                anAttribute = self.theSimulator.getStepperPropertyAttributes( aStepper, aProperty)

                if anAttribute[0] == 0:
                    pass
                
                elif aValue == '':
                    pass
                
                else:
                    aValueList = list()
                    if type( aValue ) != tuple:
                        aValueList.append( str(aValue) )
                    else:
                        aValueList = aValue

                    self.anEml.setStepperProperty( aStepper,\
                                                      aProperty,\
                                                      aValueList )
    
    def __saveEntity( self, aSystemPath='/' ):

        aVariableList = self.getEntityList(  2, aSystemPath )
        aProcessList   = self.getEntityList( 3, aSystemPath )
        aSubSystemList = self.getEntityList( 4, aSystemPath )
        
        self.__saveEntityList( 'System',   aSystemPath, aSubSystemList )
        self.__saveEntityList( 'Variable', aSystemPath, aVariableList )
        self.__saveEntityList( 'Process',  aSystemPath, aProcessList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveEntity( aSubSystemPath )
            
    def __saveEntityList( self, anEntityTypeString,\
                          aSystemPath, anIDList ):

       for anID in anIDList:
           
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = self.theSimulator.getEntityClassName( aFullID )

            if aClassName == 'System::/':
                pass
            else:
                self.anEml.createEntity( aClassName, aFullID )
            
    def __saveProperty( self, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = self.theSimulator.getEntityList( 2,  aSystemPath )
        aProcessList   = self.theSimulator.getEntityList( 3,  aSystemPath )
        aSubSystemList = self.theSimulator.getEntityList( 4,  aSystemPath )

        self.__savePropertyList( 'Variable',\
                                 aSystemPath, aVariableList )
        self.__savePropertyList( 'Process',  aSystemPath, aProcessList )
        self.__savePropertyList( 'System',\
                                 aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveProperty( aSubSystemPath )

    def __savePropertyList( self, anEntityTypeString,\
                            aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = self.theSimulator.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                
                aValue = self.theSimulator.getEntityProperty(aFullPN)
                anAttribute = self.theSimulator.getEntityPropertyAttributes(aFullPN)

                #anExclusionList = ('Concentration',
                #                   'Fixed',
                #                   'FullID',
                #                   'Name',
                #                   'TotalVelocity',
                #                   'Velocity',
                #                   'Activity',
                #                   'Priority',
                #                   )
                
                if anAttribute[0] == 0:
                    pass
                
                #elif aProperty in anExclusionList:
                #    pass
                
                elif aValue == '':
                    pass
                
                else:
                    
                    aValueList = list()
                    if type( aValue ) != tuple:
                        aValueList.append( str(aValue) )
                    else:
                        # ValueList convert into string for eml
                        aValueList = self.__convertPropertyValueList( aValue )
                        #aValueList = aValue
                        
                    self.anEml.setEntityProperty( aFullID, aProperty, aValueList )
                    
    def __convertPropertyValueList( self, aValueList ):
        
        aList = list()

        for aValueListNode in aValueList:

            if type( aValueListNode ) == tuple:

                if type( aValueListNode[0] ) == tuple:
                    aConvertedList = self.__convertPropertyValueList( aValueListNode )
                else:
                    aConvertedList = map(str, aValueListNode)

                aList.append( aConvertedList )

        return aList

    class _session_prompt:
        def __init__( self, aSession ):
            self.theSession = aSession

        def __str__( self ):
            if self.theSession.theModelName == '':
                return 'ecell3-session>>> '
            else:
                return '<%s, t=%g>>> ' %\
                       ( self.theSession.theModelName,\
                         self.theSession.getCurrentTime() )

if __name__ == "__main__":
    pass
