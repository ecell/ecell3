#! /usr/bin/env python

import string
import eml
import sys
import os
import time


from Numeric import *
import ecell.emc
import ecell.ecs

from ecell.ecssupport import *
from ecell.DataFileManager import *
from ecell.ECDDataFile import *

#from ecell.FullID import *
#from ecell.util import *
#from ecell.ECS import *

BANNERSTRING =\
'''ecell3-session [ E-Cell SE Version %s, on Python Version %d.%d.%d ]
Copyright (C) 1996-2003 Keio University.
Send feedback to Kouichi Takahashi <shafi@e-cell.org>'''\
% ( ecell.ecs.getLibECSVersion(), sys.version_info[0], sys.version_info[1], sys.version_info[2] )


class Session:
    'Session class'


    def __init__( self, aSimulator=None ):
        'constructor'

        self.theMessageMethod = self.__plainMessageMethod

        if aSimulator is None:
            self.theSimulator = ecell.emc.Simulator()
        else:
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
        anEml = eml.Eml()

        # creates root entity
        anEml.createEntity('CompartmentSystem', 'System::/')
        
        # calls save methods
        self.__saveEntity( anEml )
        self.__saveStepper( anEml )
        self.__saveProperty( anEml )

        # if the type is string
        if type( aModel ) == str:

            # add comment
            aCurrentInfo = '''<!-- created by ecell.Session.saveModel
 date: %s
 currenttime: %s
-->
<eml>
''' % ( time.asctime( time.localtime() ) , self.getCurrentTime() )
       	    aString = anEml.asString()
            aBuffer = string.join( string.split(aString, '<eml>\n'), aCurrentInfo)
            aFileObject = open( aModel, 'w' )
            aFileObject.write( aBuffer )
            aFileObject.close()

        # if the type is file object
        elif type( aModel ) == file:
       	    aString = anEml.asString()
            aFileObject = aModel
            aFileObject.write( aString )
            aFileObject.close()

        # When the type doesn't match
        else:
            raise TypeError, " The type of aModel must be string(file name) or file object "

    # end of saveModel


    def restoreMessageMethod(self):
	self.theMessageMethod=self.__plainMessageMethod
        
    def setMessageMethod( self, aMethod ):
        self.theMessageMethod = aMethod

    def message( self, message ):
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

    def saveLoggerData( self, fullpn=0, aSaveDirectory='./Data', aStartTime=-1, anEndTime=-1, anInterval=-1 ):
        
        # -------------------------------------------------
        # Check type.
        # -------------------------------------------------
        
        aLoggerNameList = []

        if type( fullpn ) == str:
            aLoggerNameList.append( aFullPNString )
        elif not fullpn :
            aLoggerNameList = self.getLoggerList()
        elif type( fullpn ) == list: 
            aLoggerNameList = fullpn
        elif type( fullpn ) == tuple: 
            aLoggerNameList = fullpn
        else:
            self.mesage( fullpn +" is not suitable type.\nuse string or list or tuple" )
            sys.exit(0)
            
        # -------------------------------------------------
        # Execute saving.
        # -------------------------------------------------
        if not os.path.isdir( aSaveDirectory ):
            os.mkdir( aSaveDirectory )

        # creates instance datafilemanager
        aDataFileManager = DataFileManager()

        # sets root directory to datafilemanager
        aDataFileManager.setRootDirectory(aSaveDirectory)
        
        aFileIndex=0

        try: #(1)
            
            # gets all list of selected property name
            for aFullPNString in aLoggerNameList: #(2)

                # -------------------------------------------------
                # from [Variable:/CELL/CYTOPLASM:E:Value]
                # to   [Variable_CELL_CYTOPLASM_E_Value]
                # -------------------------------------------------

                aRootIndex=find(aFullPNString,':/')
                aFileName=aFullPNString[:aRootIndex]+aFullPNString[aRootIndex+1:]
                aFileName=replace(aFileName,':','_')
                aFileName=replace(aFileName,'/','_')
                
                aECDDataFile = ECDDataFile()
                aECDDataFile.setFileName(aFileName)
                
                # -------------------------------------------------
                # Gets logger
                # -------------------------------------------------
                # need check if the logger exists
                aLoggerStub = self.createLoggerStub( aFullPNString )
                if not aLoggerStub.isExist():
                    aErrorMessage='\nLogger doesn\'t exist.!\n'
                    self.message( aErrorMessage )
                    return None
                aLoggerStartTime= aLoggerStub.getStartTime()
                aLoggerEndTime= aLoggerStub.getEndTime()
                if aStartTime == -1 or anEndTime == -1:
                    # gets start time and end time from logger
                    aStartTime = aLoggerStartTime
                    anEndTime = aLoggerEndTime
                else:
                    # checks the value
                    if not ( aLoggerStartTime < aStartTime < aLoggerEndTime ):
                        aStartTime = aLoggerStartTime
                    if not ( aLoggerStartTime < anEndTime < aLoggerEndTime ):
                        anEndTime = aLoggerEndTime

                # -------------------------------------------------
                # gets the matrix data from logger.
                # -------------------------------------------------
                if anInterval == -1:
                    # gets data with specifing interval 
                    aMatrixData = aLoggerStub.getData( aStartTime, anEndTime )
                else:
                    # gets data without specifing interval 
                    aMatrixData = aLoggerStub.getData( aStartTime, anEndTime, anInterval )

                    
                # sets data name 
                aECDDataFile.setDataName(aFullPNString)

                # sets matrix data
                aECDDataFile.setData(aMatrixData)

                # -------------------------------------------------
                # adds data file to data file manager
                # -------------------------------------------------
                aDataFileManager.getFileMap()[`aFileIndex`] = aECDDataFile
                
                aFileIndex = aFileIndex + 1

                # for(2)
                
            aDataFileManager.saveAll()

        except: #try(1)

            # -------------------------------------------------
            # displays error message and exit this method.
            # -------------------------------------------------

            import sys
            import traceback 
            print __name__,
            aErrorMessageList = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
            for aLine in aErrorMessageList: 
                self.message( aLine ) 
                
            aErrorMessage= "Error : could not save [%s] " %aFullPNString
            self.message( aErrorMessage )
            sys.exit(0)

        else: # try(1)
            # -------------------------------------------------
            # displays error message and exit this method.
            # -------------------------------------------------
            
            aSuccessMessage= " All files you selected are saved. " 
            self.message( aSuccessMessage )

        # end of try(1)

	# end of saveData

    def plainMessageMethod( self, aMessage ):
	self.__plainMessageMethod( aMessage )

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

            aFullPNList = map( lambda x: aFullID + ':' + x, aPropertyList ) 
            aValueList = map( anEml.getEntityProperty, aFullPNList )
            map( self.theSimulator.setEntityProperty, aFullPNList, aValueList )

            
    def __loadEntityList( self, anEml, anEntityTypeString,\
                          aSystemPath, anIDList ):
        
        aPrefix = anEntityTypeString + ':' + aSystemPath + ':'

        aFullIDList = map( lambda x: aPrefix + x, anIDList )
        aClassNameList = map( anEml.getEntityClass, aFullIDList )
        map( self.theSimulator.createEntity, aClassNameList, aFullIDList )



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

    def __saveStepper( self , anEml ):
        """stepper loader"""

        aStepperList = self.theSimulator.getStepperList()

        for aStepper in aStepperList:

            aClassName = self.theSimulator.getStepperClassName( aStepper )
            anEml.createStepper( str( aClassName ),\
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

                    anEml.setStepperProperty( aStepper,\
                                                      aProperty,\
                                                      aValueList )
    
    def __saveEntity( self, anEml, aSystemPath='/' ):

        aVariableList = self.getEntityList(  'Variable', aSystemPath )
        aProcessList   = self.getEntityList( 'Process', aSystemPath )
        aSubSystemList = self.getEntityList( 'System', aSystemPath )
        
        self.__saveEntityList( anEml, 'System',   aSystemPath, aSubSystemList )
        self.__saveEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__saveEntityList( anEml, 'Process',  aSystemPath, aProcessList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveEntity( anEml, aSubSystemPath )
            
    def __saveEntityList( self, anEml, anEntityTypeString,\
                          aSystemPath, anIDList ):

       for anID in anIDList:
           
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aClassName = self.theSimulator.getEntityClassName( aFullID )

            if aClassName == 'System::/':
                pass
            else:
                anEml.createEntity( aClassName, aFullID )
            
    def __saveProperty( self, anEml, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = self.theSimulator.getEntityList( 'Variable',\
                                                          aSystemPath )
        aProcessList   = self.theSimulator.getEntityList( 'Process',\
                                                          aSystemPath )
        aSubSystemList = self.theSimulator.getEntityList( 'System',\
                                                          aSystemPath )

        self.__savePropertyList( anEml, 'Variable', \
                                 aSystemPath, aVariableList )
        self.__savePropertyList( anEml, 'Process', \
                                 aSystemPath, aProcessList )
        self.__savePropertyList( anEml, 'System', \
                                 aSystemPath, aSubSystemList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveProperty( anEml, aSubSystemPath )

    def __savePropertyList( self, anEml, anEntityTypeString,\
                            aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = self.theSimulator.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                
                aValue = self.theSimulator.getEntityProperty(aFullPN)
                anAttribute = self.theSimulator.getEntityPropertyAttributes(aFullPN)

                if anAttribute[0] == 0:
                    pass
                
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
                        
                    anEml.setEntityProperty( aFullID, aProperty, aValueList )
                    
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
