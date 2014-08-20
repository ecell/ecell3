#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import eml
import sys
import os
import time


from numpy import *
import ecell.ecs
import ecell.config as config

from ecell.ecssupport import *
from ecell.emparser import Preprocessor, convertEm2Eml
from ecell.DataFileManager import *
from ecell.ECDDataFile import *

__all__ = (
    'Session',
    )

class Session:
    '''Session class'''

    def __init__( self, aSimulator=None ):
        'constructor'

        self.theMessageMethod = self.__plainMessageMethod

        if aSimulator is None:
            aSimulator = ecell.ecs.Simulator()
            aSimulator.setDMSearchPath(
                aSimulator.DM_SEARCH_PATH_SEPARATOR.join( config.dm_path ) )

        self.theSimulator = aSimulator

        self.theModelName = ''

    def loadModel( self, aModel ):
        # aModel : an EML instance, a file name (string) or a file object
        # return -> None
        # This method can thwor exceptions. 

        # checks the type of aModel

        if isinstance( aModel, eml.Eml ):
            # if the type is EML instance
            anEml = aModel
            aModelName = '<eml.Eml>'  # what should this be?
        elif isinstance( aModel, str ) or isinstance( aModel, unicode ):
            # if the type is string
            _, ext = os.path.splitext( aModel )
            ext = ext.lower()
            if ext == '.eml':
                aFileObject = open( aModel )
                anEml = eml.Eml( aFileObject )
            elif ext == '.em':
                aPreprocessor = Preprocessor( open( aModel ), aModel )
                anEmFile = aPreprocessor.preprocess()
                anEmFile.seek( 0 )
                anEml = convertEm2Eml( anEmFile, False )
            else:
                raise Exception( "Unsupported file type: %s" % ext )
            aModelName = aModel
        elif isinstance( aModel, file ):
            # change directory to file's home directory
            # if the type is file object
            aFileObject = aModel
            aModelName = aModel.name
            anEml = eml.Eml( aFileObject )
        else:
            # When the type doesn't match
            raise TypeError, "The type of aModel must be EML instance, string(file name) or file object "
    
        # calls load methods
        self.__loadStepper( anEml )
        self.__loadEntity( anEml )
        self.__loadAllProperty( anEml )
        self.theSimulator.initialize()

        # saves ModelName 
        self.theModelName = aModelName

    # end of loadModel
        

    def saveModel( self , aModel ):
        # aModel : a file name (string) or a file object
        # return -> None
        # This method can thwor exceptions. 
        
        # creates ana seve an EML instance 
        anEml = eml.Eml()

        # calls save methods
        self.__saveAllStepper( anEml )
        self.__saveEntity( anEml, 'System::/' )
        self.__saveAllEntity( anEml )
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
            aBuffer = aCurrentInfo.join( aString.split( '<eml>\n' ) )
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
            raise TypeError, "The type of aModel must be string(file name) or file object "

    # end of saveModel
    
    def restoreMessageMethod( self ):
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
#        def __dir__( self ):
#            pass
#        
        return EntityStub( self.theSimulator, fullid )

    def getEntityProperty( self, fullPN ):
        return self.theSimulator.getEntityProperty( fullPN )

    def getEntityPropertyAttributes( self, fullPN ):
        return self.theSimulator.getEntityPropertyAttributes( fullPN )

    def setEntityProperty( self, fullPN, aValue ):
        self.theSimulator.setEntityProperty( fullPN, aValue )

    def getSystemPathList( self ):
        targets = [ '/' ]
        SystemPaths = []
        while len( targets ):
            aSystemPath = targets.pop()
            SystemPaths.append( aSystemPath )
            
            subSystemIDs = self.getEntityList( 'System', aSystemPath )
            if len( subSystemIDs ):
                if aSystemPath == '/':
                    aSystemPath = ''
                targets.extend( [ '/'.join( [ aSystemPath, s ] ) for s in subSystemIDs ] )
        
        return SystemPaths    # The first element is the root System

    def getSystemList( self ):
        SystemList = [ 'System::/' ]
        SystemPaths = [ path[ 1: ] for path in self.getSystemPathList() ]     # '/A/B' -> 'A/B'
        SystemPaths.pop( 0 )
        
        for path in SystemPaths:
            path_list = path.split( '/' )
            SystemList.append( 'System:/{path}:{ID}'.format( ID = path_list.pop(), path = '/'.join( path_list ) ) )
        
        return SystemList

    def getVariableList( self ):
        VariableList = []
        for vl in [ [ path, self.getEntityList( 'Variable', path ) ] for path in self.getSystemPathList() ]:
            VariableList.extend( [ 'Variable:{path}:{ID}'.format( path = vl[ 0 ], ID = v ) for v in vl[ 1 ] ] )
        
        return VariableList

    def getProcessList( self ):
        ProcessList = []
        for pl in [ [ path, self.getEntityList( 'Process', path ) ] for path in self.getSystemPathList() ]:
            ProcessList.extend( [ 'Process:{path}:{ID}'.format( path = pl[ 0 ], ID = p ) for p in pl[ 1 ] ] )
        
        return ProcessList

    def getModelEntityList( self ):
        return self.getSystemList() + self.getVariableList() + self.getProcessList()


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
            aLoggerNameList.append( fullpn )
        elif not fullpn :
            aLoggerNameList = self.getLoggerList()
        elif type( fullpn ) == list: 
            aLoggerNameList = fullpn
        elif type( fullpn ) == tuple: 
            aLoggerNameList = fullpn
        else:
            self.message( "%s is not suitable type.\nuse string or list or tuple"%fullpn )
            return
            
        # -------------------------------------------------
        # Execute saving.
        # -------------------------------------------------
        if not os.path.isdir( aSaveDirectory ):
           os.mkdir( aSaveDirectory )

        # creates instance datafilemanager
        aDataFileManager = DataFileManager()

        # sets root directory to datafilemanager
        aDataFileManager.setRootDirectory( aSaveDirectory )
        
        aFileIndex=0

            
            # gets all list of selected property name
        for aFullPNString in aLoggerNameList: #(2)

             # -------------------------------------------------
            # from [Variable:/CELL/CYTOPLASM:E:Value]
            # to   [Variable_CELL_CYTOPLASM_E_Value]
             # -------------------------------------------------

            aRootIndex = aFullPNString.find( ':/' )
            aFileName = aFullPNString[:aRootIndex]+aFullPNString[aRootIndex+1:]
            aFileName = aFileName.replace( ':', '_' )
            aFileName = aFileName.replace( '/', '_' )
            
            aECDDataFile = ECDDataFile()
            aECDDataFile.setFileName( aFileName )
            
            # -------------------------------------------------
            # Gets logger
            # -------------------------------------------------
            # need check if the logger exists
            aLoggerStub = self.createLoggerStub( aFullPNString )
            if not aLoggerStub.exists():
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

        try: #(1)
                
            aDataFileManager.saveAll()

        except: #try(1)

            # -------------------------------------------------
            # displays error message and exit this method.
            # -------------------------------------------------

            import traceback 
            print __name__,
            aErrorMessageList = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
            for aLine in aErrorMessageList: 
                self.message( aLine ) 
                
            aErrorMessage= "Error : could not save [%s] " %aFullPNString
            self.message( aErrorMessage )


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

            try:
                self.theSimulator.createStepper( str( aClassName ),\
                                                 str( aStepper ) )
            except RuntimeError, e:
                raise RuntimeError( 'Failed to create Stepper [%s]: ' % (aStepper,) +\
                                    str( e ) )

            aPropertyList = anEml.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                aValue = anEml.getStepperProperty( aStepper, aProperty )

                try:
                    self.theSimulator.loadStepperProperty( aStepper,\
                                                           aProperty,\
                                                           aValue )
                except RuntimeError, e:
                    raise RuntimeError( 'When creating Stepper [%s], ' % (aStepper,) +\
                                        'failed to set property [%s]: ' % (aProperty,) +\
                                        str( e ) )

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


    def __loadAllProperty( self, anEml, aSystemPath='' ):
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
            self.__loadAllProperty( anEml, aSubSystemPath )

    def __loadPropertyList( self, anEml, anEntityTypeString,\
                            aSystemPath, anIDList ):

        for anID in anIDList:
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = anEml.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:                
                aFullPN = aFullID + ':' + aProperty
                aValue = anEml.getEntityProperty( aFullPN )
                try:
                    self.theSimulator.loadEntityProperty( aFullPN, aValue )
                except RuntimeError, e:
                    raise RuntimeError( 'Failed to set Entity property [%s],'
                                        % aFullPN \
                                        + 'value =:\n%s\n' % str( aValue ) +\
                                        str( e ) )

    def __loadEntityList( self, anEml, anEntityTypeString,\
                          aSystemPath, anIDList ):
        
        aPrefix = anEntityTypeString + ':' + aSystemPath + ':'

        for anID in anIDList:
            aClassName = anEml.getEntityClass( aPrefix + anID )
            aFullID = aPrefix + anID
            
            try:
                self.theSimulator.createEntity( aClassName, aFullID )
            except RuntimeError, e:
                raise RuntimeError( 'Failed to create Entity [%s]: ' % (aFullID,) +\
                                    str( e ) )



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

    def __saveAllStepper( self , anEml ):
        """stepper loader"""

        aStepperList = self.theSimulator.getStepperList()

        for aStepper in aStepperList:

            aClassName = self.theSimulator.getStepperClassName( aStepper )
            anEml.createStepper( str( aClassName ), \
                                             str( aStepper ) )

            aPropertyList = self.theSimulator.getStepperPropertyList( aStepper )

            for aProperty in aPropertyList:
                
                anAttributeList = self.theSimulator.getStepperPropertyAttributes( aStepper, aProperty )

                # check get attribute 
                if anAttributeList[3] != 0:
                                    
                    aValue = self.theSimulator.\
                            getStepperProperty( aStepper, aProperty )
                        
                    #if aValue == '':
                    #    pass
                    
                    aValueList = list()
                    if type( aValue ) != tuple:
                        aValueList.append( str( aValue ) )
                    else:
                        aValueList = aValue

                    anEml.setStepperProperty( aStepper, \
                                              aProperty, \
                                              aValueList )
    
    def __saveAllEntity( self, anEml, aSystemPath='/' ):

        aVariableList = self.getEntityList( 'Variable', aSystemPath )
        aProcessList   = self.getEntityList( 'Process', aSystemPath )
        aSubSystemList = self.getEntityList( 'System', aSystemPath )
        
        self.__saveEntityList( anEml, 'System', aSystemPath, aSubSystemList )
        self.__saveEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__saveEntityList( anEml, 'Process', aSystemPath, aProcessList )

        for aSystem in aSubSystemList:
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            self.__saveAllEntity( anEml, aSubSystemPath )
            
    def __saveEntityList( self, anEml, anEntityTypeString, \
                          aSystemPath, anIDList ):

       for anID in anIDList:
           
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            self.__saveEntity( anEml, aFullID )

    def __saveEntity( self, anEml, aFullID ):
        aClassName = self.theSimulator.getEntityClassName( aFullID )
        anEml.createEntity( aClassName, aFullID )
            

    def __saveProperty( self, anEml, aSystemPath='' ):
        # the default of aSystemPath is empty because
        # unlike __loadEntity() this starts with the root system

        aVariableList  = self.theSimulator.getEntityList( 'Variable', \
                                                          aSystemPath )
        aProcessList   = self.theSimulator.getEntityList( 'Process', \
                                                          aSystemPath )
        aSubSystemList = self.theSimulator.getEntityList( 'System', \
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


    def __savePropertyList( self, anEml, anEntityTypeString, \
                            aSystemPath, anIDList ):

        for anID in anIDList:

            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = self.theSimulator.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                
                anAttributeList = self.theSimulator.getEntityPropertyAttributes( aFullPN )

                # check savable
                if anAttributeList[3] != 0:
                    
                    aValue = self.theSimulator.saveEntityProperty( aFullPN )
                    #print aValue

                    if aValue != '':

                        aValueList = list()
                        if type( aValue ) != tuple:
                            aValueList.append( str( aValue ) )
                            
                        elif aValue == ():
                            # exclude the empty tuple (ad-hoc, Jul. 21, 2004)
                            break
                        
                        else:
                            # ValueList convert into string for eml
                            aValueList = self.__convertPropertyValueList( aValue )

                            #aValueList = aValue
                            
                        anEml.setEntityProperty( aFullID, aProperty, 
                                                 aValueList )
 
    def __convertPropertyValueList( self, aValueList ):
       
        aList = list()
        tmpList = list()

        for aValueListNode in aValueList:

            if type( aValueListNode ) == tuple:
                # for recursive values

                if type( aValueListNode[0] ) == tuple:
                    aConvertedList = self.__convertPropertyValueList( aValueListNode )
                else:
                    aConvertedList = map( str, aValueListNode )
                    
                aList.append( aConvertedList )
                
            else:
                # for the 1st level tuple (not for the recursive)
                tmpList.append( aValueListNode )

        if tmpList != []:
            aList.append( tmpList )
        else:
            pass


        return aList
    


def createScriptContext( session, parameters ):
    # theSession == self in the script
    aContext = { 'theSession': session, 'self': session }

    # flatten class methods and object properties so that
    # 'self.' isn't needed for each method calls in the script
    aKeyList = list( session.__dict__.keys() +\
                     session.__class__.__dict__.keys() )
    aDict = {}
    for aKey in aKeyList:
        if not aKey.startswith('__'):
            aDict[ aKey ] = getattr( session, aKey )

    aContext.update( aDict )
        
    # add parameters to the context
    aContext.update( parameters )

    return aContext


if __name__ == "__main__":
    pass
