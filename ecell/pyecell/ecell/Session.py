#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
from decimal import Decimal
from numpy import *
from warnings import *

import ecell.identifiers as identifiers
import ecell.ecs as ecs
import ecell.emc as emc
from ecell.EntityStub import EntityStub

from ecell.ecssupport import *
from ecell.DataFileManager import DataFileManager
from ecell.ECDDataFile import ECDDataFile

import ecell.util as util

__all__ = (
    'Session'
    )

class Session:
    'Session class'

    def __init__( self, aSimulator = None ):
        'constructor'

        self.theMessageMethod = self.__plainMessageMethod

        if aSimulator is None:
            self.theSimulator = emc.Simulator()
        else:
            self.theSimulator = aSimulator

        self.theModelName = ''

    def loadModel( self, aModel ):
        # aModel : an EML instance, a file name (string) or a file object
        # return -> None
        # This method may throw an exception. 

        # checks the type of aModel

        # if the type is EML instance
        if type( aModel ) == type( eml.Eml ) or\
               type( aModel ) == type( eml.Eml() ):
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
    
        # change directory to file's home directory
        if type( aModel ) != type( eml.Eml ) and\
               type( aModel ) != type( eml.Eml() ):
            dirname = os.path.dirname( aModel )
            if dirname != "":
                os.chdir( dirname )
    
        # calls load methods
        self.__loadStepper( anEml )
        self.__loadEntity( anEml )
        self.__loadAllProperty( anEml )

        # saves ModelName 
        self.theModelName = aModelName

    def isModelLoaded( self ):
        return len( self.theModelName ) > 0

    def saveModel( self , aModel ):
        # aModel : a file name (string) or a file object
        # return -> None
        # This method can thwor exceptions. 
        
        # creates ana seve an EML instance 
        anEml = eml.Eml()

        # calls save methods
        self.__saveAllStepper( anEml )
        self.__saveEntity( anEml, 'System:/:' )
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

    def restoreMessageMethod( self ):
        self.theMessageMethod=self.__plainMessageMethod
        
    def setMessageMethod( self, aMethod ):
        self.theMessageMethod = aMethod

    def message( self, message ):
        self.theMessageMethod( message )

    # Simulator methods
    def run( self, time = None ):
        if not time:
            self.theSimulator.run()
        else:
            self.theSimulator.run( time )

    def stop( self ):
        self.theSimulator.stop()

    def step( self, num = 1 ):
        self.theSimulator.step( num )

    def getNextEvent( self ):
        return self.theSimulator.getNextEvent()

    def getCurrentTime( self ):
        return self.theSimulator.getCurrentTime()

    def setEventChecker( self, event ):
        self.theSimulator.setEventChecker( event )

    def setEventHandler( self, event ):
        self.theSimulator.setEventHandler( event )

    # Stepper methods
    def getStepperList( self ):
        return self.theSimulator.getStepperList()

    def createStepperStub( self, id ):
        return StepperStub( self.theSimulator, id )

    # Entity methods
    def getEntityList( self, entityType, systemPath ):
        if type( entityType ) == int or type( entityType ) == Decimal:
            entityType = ENTITYTYPE_LIST[ entityType ]
        return self.theSimulator.getEntityList( entityType, str( systemPath or '' ) )

    def createEntityStub( self, fullid ):
        return EntityStub( self.theSimulator, fullid )

    # Logger methods
    def getLoggerList( self ):
        return self.theSimulator.getLoggerList()

    def getLoggedPNList( self ):
        return [ identifiers.FullPN( aFullPNString )
            for aFullPNString in self.theSimulator.getLoggerList() ]
        
    def createLogger( self, fullpn ):
        warning( 'Use LoggerStub instead', DeprecationWarning, stacklevel = 2 )
        aStub = self.createLoggerStub( fullpn )
        aStub.create()

    def createLoggerStub( self, fullpn ):
        return LoggerStub( self.theSimulator, fullpn )

    def saveLoggerData( self, anOpaqueData, aSaveDirectory='./Data', aStartTime = -1, anEndTime = -1, anInterval = -1 ):
        if anOpaqueData == None:
            aLoggerNameList = self.getLoggerList()
        elif type( anOpaqueData ) == str or type( anOpaqueData ) == unicode:
            aLoggerNameList = [ anOpaqueData ]
        else:
            aLoggerNameList = anOpaqueData
        aLoggerNameList = map( str, aLoggerNameList )

        if not os.path.isdir( aSaveDirectory ):
            try:
                os.mkdir( aSaveDirectory )
            except:
                self.message( "Failed to create %s." % aSaveDirectory )
                return

        # creates instance datafilemanager
        aDataFileManager = DataFileManager()

        # sets root directory to datafilemanager
        aDataFileManager.setRootDirectory( aSaveDirectory )
        
        aFileIndex=0
            
        # gets all list of selected property name
        for aFullPNString in aLoggerNameList: #(2)
            # from [Variable:/CELL/CYTOPLASM:E:Value]
            # to   [Variable_CELL_CYTOPLASM_E_Value]
            aRootIndex=find( aFullPNString, ':/' )
            aFileName=aFullPNString[:aRootIndex]+aFullPNString[aRootIndex+1:]
            aFileName=replace( aFileName, ':', '_' )
            aFileName=replace( aFileName, '/', '_' )
            
            aECDDataFile = ECDDataFile()
            aECDDataFile.setFileName( aFileName )
            
            # Gets logger
            # need check if the logger exists
            aLoggerStub = self.createLoggerStub( aFullPNString )
            if not aLoggerStub.exists():
                self.message( "Logger does not exist!" )
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

            # gets the matrix data from logger.
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

            # adds data file to data file manager
            aDataFileManager.getFileMap()[`aFileIndex`] = aECDDataFile
            
            aFileIndex = aFileIndex + 1

            # for(2)

        try: #(1)
            aDataFileManager.saveAll()
        except: #try(1)
            # displays error message and exit this method.
            import traceback 
            print __name__,
            aErrorMessageList = traceback.format_exception(sys.exc_type,sys.exc_value,sys.exc_traceback)
            for aLine in aErrorMessageList: 
                self.message( aLine ) 
                
            aErrorMessage = "Error : could not save [%s] " %aFullPNString
            self.message( aErrorMessage )


        else: # try(1)
            # -------------------------------------------------
            # displays error message and exit this method.
            # -------------------------------------------------
            
            aSuccessMessage = " All files you selected are saved. " 
            self.message( aSuccessMessage )

        # end of try(1)

    def plainMessageMethod( self, aMessage ):
        self.__plainMessageMethod( aMessage )

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

    def __loadEntity( self, anEml, aSystemPath = None ):
        if aSystemPath == None:
            self.__loadEntity( anEml, identifiers.SystemPath( "/" ) )
            return
        assert isinstance( aSystemPath, identifiers.SystemPath )
        aVariableList = anEml.getEntityList( 'Variable', aSystemPath )
        aProcessList   = anEml.getEntityList( 'Process', aSystemPath )
        aSubSystemList = anEml.getEntityList( 'System',  aSystemPath )

        self.__loadEntityList( anEml, 'Variable', aSystemPath, aVariableList )
        self.__loadEntityList( anEml, 'Process',  aSystemPath, aProcessList )
        self.__loadEntityList( anEml, 'System',   aSystemPath, aSubSystemList )

        for anID in aSubSystemList:
            self.__loadEntity( anEml, identifiers.SystemPath( aSystemPath, anID ) )

    def __loadAllProperty( self, anEml, aSystemPath = None ):
        if aSystemPath == None:
            self.__loadPropertyList( anEml, 'System', '/', [ "" ] )
            self.__loadAllProperty( anEml, identifiers.SystemPath( "/" ) )
            return
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

        for anID in aSubSystemList:
            self.__loadAllProperty( anEml,
                identifiers.SystemPath( aSystemPath, anID ) )

    def __loadPropertyList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):

        for anID in anIDList:
            aFullID = identifiers.FullID(
                anEntityTypeString, aSystemPath, anID )
            aPropertyList = anEml.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:                
                aFullPN = identifiers.FullPN( aFullID, aProperty )
                aValue = anEml.getEntityProperty( aFullPN )
                try:
                    # Enclose it if not a list
                    if type( aValue ) != list:
                        aValue = [ aValue ]
                    # XXX: Unicode to native conversion
                    aValue = util.toNative( aValue )
                    self.theSimulator.loadEntityProperty(
                        str( aFullPN ), aValue )
                except RuntimeError, e:
                    raise RuntimeError( 'Failed to set Entity property [%s],'
                                        % aFullPN \
                                        + 'value =:\n%s\n' % str( aValue ) +\
                                        str( e ) )

    def __loadEntityList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):
        for anID in anIDList:
            aFullID = identifiers.FullID(
                anEntityTypeString, aSystemPath, anID )
            aClassName = anEml.getEntityClass( aFullID )

            try:
                self.theSimulator.createEntity( aClassName, str( aFullID ) )
            except RuntimeError, e:
                raise RuntimeError( 'Failed to create Entity [%s]: ' % (aFullID,) +\
                                    str( e ) )

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
            
    def __saveEntityList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):
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

    def __savePropertyList( self, anEml, anEntityTypeString, aSystemPath, anIDList ):
        for anID in anIDList:
            aFullID = anEntityTypeString + ':' + aSystemPath + ':' + anID
            aPropertyList = self.theSimulator.getEntityPropertyList( aFullID )

            for aProperty in aPropertyList:
                aFullPN = aFullID + ':' + aProperty
                
                anAttributeList = self.theSimulator.getEntityPropertyAttributes( aFullPN )
                # check savable
                if anAttributeList[3] != 0:
                    aValue = self.theSimulator.saveEntityProperty( aFullPN )
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

        return aList
 
