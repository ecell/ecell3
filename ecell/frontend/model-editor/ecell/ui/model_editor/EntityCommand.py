#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2015 Keio University
#       Copyright (C) 2008-2015 RIKEN
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

import gtk
import gobject

from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.DMInfo import *
from ecell.ui.model_editor.Command import *
from ecell.ui.model_editor.StepperChooser import *

__all__ = (
    'CreateEntity',
    'DeleteEntityList',
    'RenameEntity',
    'CopyEntityList',
    'CutEntityList',
    'PasteEntityList',
    'ChangeEntityClass',
    'ChangeEntityProperty',
    'CreateEntityProperty',
    'DeleteEntityPropertyList',
    'RenameEntityProperty',
    'CopyEntityPropertyList',
    'CutEntityPropertyList',
    'PasteEntityPropertyList',
    'SetEntityInfo',
    'RelocateEntity',
    )

class CreateEntity(ModelCommand):
    """
    arg1: ID
    arg2: ClassName eg. CompartmentSystem
    """
    ARGS_NO = 2
    FULLID = 0
    CLASSNAME = 1
    
    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        
        self.__theID = self.theArgs[ self.FULLID ]
        self.__theClassName = self.theArgs[ self.CLASSNAME ]
        # CHECK IN DM WHETHER CLASSNAME EXISTS!!!
        if self.theModel.isEntityExist(self.__theID):
            return False
        if not isFullIDEligible( self.__theID ):
            return False
        return True
    
    def do( self ):

        self.theModel.createEntity( self.__theClassName, self.__theID )
        anEntityType = self.__theID.split( ':' )[0]
        if anEntityType == ME_SYSTEM_TYPE:
            aParentFullID = convertSysPathToSysID(  self.__theID.split( ':' )[1]  )
            parentStepperFullPN = createFullPN( aParentFullID , MS_SYSTEM_STEPPERID )
            aStepperIdValue = self.theModel.getEntityProperty(parentStepperFullPN)
            if aStepperIdValue != '': 
                aFullPN = createFullPN( self.__theID, MS_SYSTEM_STEPPERID )
                self.theModel.setEntityProperty( aFullPN, [aStepperIdValue])
            else:
                pass
            newFullID = ':'.join( [ ME_VARIABLE_TYPE, convertSysIDToSysPath( self.__theID ), 'SIZE' ] )
            self.theModel.createEntity( ME_VARIABLE_TYPE, newFullID )
            self.theModel.setEntityProperty( newFullID + ":Value", 1 )


        return True

    def createReverseCommand( self ):
        self.theReverseCommandList = [ DeleteEntityList( self.theReceiver, [ self.__theID ] ) ]

    def getAffected( self ):
        return (getFullIDType( self.__theID ), getParentSystemOfFullID( self.__theID ) )


class DeleteEntityList(ModelCommand):
    """
    args:
    """
    IDLIST = 0
    ARGS_NO = 1
    
    def checkArgs( self ):
        

        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]        
        self.__theType = getFullIDType( self.__theIDList[0] )
        return True
    
    def do( self ):
        

        sortedList = self.theBufferFactory.sortSystemIDs( self.__theIDList )        
        self.theReverseCommandList = []
        self.theAffectedPath = None
        for anID in sortedList:

            if not self.theModel.isEntityExist( anID ):
                continue

            tempBuffer = self.theBufferFactory.createEntityListBuffer( self.__theType, [anID] )

            self.theModel.deleteEntity( anID )


            aParentSys = convertSysIDToSysPath ( getParentSystemOfFullID( anID ) )

            if self.theAffectedPath == None:
                self.theAffectedPath = aParentSys
            else:
                self.theAffectedPath = getMinPath( self.theAffectedPath, aParentSys )
            aReverseCommand =  PasteEntityList( self.theReceiver, tempBuffer, aParentSys ) 
            self.theReverseCommandList.insert(0, aReverseCommand )
        return True
            
    def createReverseCommand( self ):
        
        #reverse commandlist is created throughout do command
        self.theReverseCommandList = None

    def getAffected( self ):
        return (self.__theType,  convertSysPathToSysID( self.theAffectedPath ) ) 


class RenameEntity(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    OLDID = 0
    NEWID = 1

    def checkArgs( self ):
        # oldID should exist
        # newID shouldn't exist
        if not ModelCommand.checkArgs(self):
            return False
        
        self.__theOldID = self.theArgs[ self.OLDID ]
        self.__theNewID = self.theArgs[ self.NEWID ]
        if not isFullIDEligible( self.__theNewID ):
            return False

        if not self.theModel.isEntityExist(self.__theOldID):
            return False
        if self.theModel.isEntityExist( self.__theNewID ):
            return False
        self.__theType = getFullIDType( self.__theOldID )
        return True

    def do( self ):
        
        # store properties
        aFullPNList = []
        for aPropertyName in self.theModel.getEntityPropertyList( self.__theOldID ):
            aFullPNList.append( createFullPN( self.__theOldID, aPropertyName ) )

        propBuffer = self.theBufferFactory.createEntityPropertyListBuffer( aFullPNList  )

        aClassName = self.theModel.getEntityClassName(self.__theOldID )
        # if it is system take systemlist, processlist, variablelist
        if self.__theType == ME_SYSTEM_TYPE:
            sysPath = convertSysIDToSysPath( self.__theOldID )
            systemNameList = self.theModel.getEntityList( ME_SYSTEM_TYPE, sysPath )
            systemList = createFullIDList( ME_SYSTEM_TYPE, self.__theOldID, systemNameList )

            processNameList = self.theModel.getEntityList( ME_PROCESS_TYPE, sysPath )
            processList = createFullIDList( ME_PROCESS_TYPE, self.__theOldID, processNameList )

            variableNameList = self.theModel.getEntityList( ME_VARIABLE_TYPE, sysPath )
            variableList = createFullIDList( ME_VARIABLE_TYPE, self.__theOldID, variableNameList )

            systemBuffer = self.theBufferFactory.createSystemListBuffer( systemList )
            processBuffer = self.theBufferFactory.createProcessListBuffer( processList )
            variableBuffer = self.theBufferFactory.createVariableListBuffer( variableList )
            
        # delete system
        self.theModel.deleteEntity( self.__theOldID )

        # create newsystem
        self.theModel.createEntity( aClassName, self.__theNewID )

        # paste property list
        self.theBufferPaster.pasteEntityPropertyListBuffer( propBuffer, self.__theNewID )

        # if system paste systemlist, processlist, variablelist
        if self.__theType == ME_SYSTEM_TYPE:
            sysPath = convertSysIDToSysPath( self.__theNewID )
            self.theBufferPaster.pasteSystemListBuffer( systemBuffer, sysPath )
            self.theBufferPaster.pasteProcessListBuffer( processBuffer, sysPath )
            self.theBufferPaster.pasteVariableListBuffer( variableBuffer, sysPath )

        
        # if variable set process variable references 
        if self.__theType == ME_VARIABLE_TYPE:
            processList = propBuffer.getProperty( ME_VARIABLE_PROCESSLIST )
            for aProcess in processList:
                # get VariableReference
                varrefFullPN = createFullPN( aProcess, ME_PROCESS_VARREFLIST )
                aVarrefList = self.theModel.getEntityProperty( varrefFullPN ) 
                # scan through varreflist
                for aVarref in aVarrefList:
                    relFlag = not isAbsoluteReference( aVarref[ME_VARREF_FULLID] )
                    aVariableID = getAbsoluteReference( aProcess, aVarref[ME_VARREF_FULLID] )
                    if aVariableID == self.__theOldID:
                        # create relativereference is necessary
                        if relFlag:
                            aVariableID = getRelativeReference( aProcess, self.__theNewID )
                        else:
                            aVariableID = self.__theNewID
                        aVarref[ME_VARREF_FULLID] = aVariableID
                # write back varreflist
                self.theModel.setEntityProperty ( varrefFullPN, aVarrefList )
        return True

    def createReverseCommand( self ):
        reversecommand = RenameEntity( self.theReceiver, self.__theNewID, self.__theOldID )
        self.theReverseCommandList = [ reversecommand ]

    def getAffected( self ):
        return (self.__theType, getParentSystemOfFullID( self.__theNewID ) )


class CopyEntityList(ModelCommand):
    """
    args:
    """

    ARGS_NO = 1
    IDLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]
        self.__theType = getFullIDType( self.__theIDList[0] )
        return True
    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createEntityListBuffer( self.__theType, self.__theIDList ) )
        return True

    def createReverseCommand( self ):
        # there is no reverse command for copy
        self.theReverseCommandList = None

    def getAffected( self ):
        return (None, None)


class CutEntityList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 1
    IDLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theIDList = self.theArgs[ self.IDLIST ]
        self.__theType = getFullIDType( self.__theIDList[0] )
        return True
    
    def do( self ):
        # create copy buffer
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createEntityListBuffer( self.__theType, self.__theIDList ) )
        self.theAffectedPath = None
        # delete list
        reverseBuffer = self.theBufferFactory.createEntityListBuffer( self.__theType, self.__theIDList )

        for anID in self.__theIDList:
            self.theModel.deleteEntity( anID )
            aParentSys = convertSysIDToSysPath ( getParentSystemOfFullID( anID ) )

            if self.theAffectedPath == None:
                self.theAffectedPath = aParentSys
            else:
                self.theAffectedPath = getMinPath( self.theAffectedPath, aParentSys )

        self.theReverseCommandList = [ PasteEntityList( self.theReceiver, reverseBuffer, aParentSys ) ]

        return True

    def createReverseCommand( self ):
        #reverse commandlist is created throughout do command
        self.theReverseCommandList = None

    def getAffected( self ):
        return (self.__theType, convertSysPathToSysID( self.theAffectedPath ) )


class PasteEntityList(ModelCommand):
    """
    args:
    """
    BUFFER = 0
    SYSPATH = 1
    ARGS_NO = 2
    
    def checkArgs( self ):

        if not ModelCommand.checkArgs(self):
            return False

        self.__theBuffer = self.theArgs[ self.BUFFER ]
        self.__theSysPath = self.theArgs[ self.SYSPATH ]
        self.__theType = self.__theBuffer.getType()

        if not self.theModel.isEntityExist( convertSysPathToSysID( self.__theSysPath ) ):
            return False

        return True

    def do( self ):
        # get entities from Buffer one by one
        entityList = self.__theBuffer.getEntityList()
        modelEntityList = self.theModel.getEntityList(  self.__theType, self.__theSysPath )
        pasteBuffer = self.theBufferFactory.createEntityListBuffer( self.__theType )
        deleteNameList = []
        if self.__theType == ME_SYSTEM_TYPE:
            entityList = self.theBufferFactory.sortSystemIDs( entityList )
        for anEntity in entityList:

            if anEntity in modelEntityList:
                # if exist ask whether to overwrite it
                aFullID = ":".join( [self.__theType, self.__theSysPath, anEntity ] )
                msgtext = aFullID + " already exists. Overwrite?"

                if self.theReceiver.printMessage( msgtext, ME_YESNO ) != ME_RESULT_CANCEL:

                    # if yes add to be destroyed to paste list
                    self.theBufferFactory.addToEntityListBuffer(  pasteBuffer, aFullID )

                    # delete old entity
                    self.theModel.deleteEntity( aFullID )
                else:
                    continue

            # add to delete buffer
            deleteNameList.append( anEntity )

            # paste it
            self.theBufferPaster.pasteEntityListBuffer(  self.__theBuffer, self.__theSysPath, anEntity )

        deleteList = createFullIDList( self.__theType, convertSysPathToSysID( self.__theSysPath), deleteNameList )
        # create a paste and a delete reverse command
        deleteCommand = DeleteEntityList( self.theReceiver, deleteList )
        pasteCommand = PasteEntityList( self.theReceiver, pasteBuffer, self.__theSysPath )
        self.theReverseCommandList = [ deleteCommand, pasteCommand ]
        return True

    def createReverseCommand( self ):
        # reverse command created through do
        self.theReverseCommandList = None

    def getAffected( self ):
        return (self.__theType, convertSysPathToSysID( self.__theSysPath ) )


class ChangeEntityClass(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    CLASSNAME = 0
    ID =1
    # SUPPORTS CHANGING CLASS IN PROCESSES ONLY!!!
    
    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.ID ]
        self.__theClassName = self.theArgs[ self.CLASSNAME ]
        if not self.theModel.isEntityExist( self.__theID ):
            return False
        self.__theType = getFullIDType( self.__theID ) 
        if self.__theType != ME_PROCESS_TYPE:
            return False
        return True

    def getAffected( self ):
        return (self.__theType, self.__theID )
    
    def do( self ):
        coreFullPNList = []
        optionalFullPNList = []
        for aPropertyName in self.theModel.getEntityPropertyList( self.__theID ):
            aFullPN = createFullPN( self.__theID, aPropertyName )
            if self.theModel.getEntityPropertyAttributes( aFullPN )[MS_DELETEABLE_FLAG ]:
                optionalFullPNList.append( aFullPN )
            else:
                coreFullPNList.append( aFullPN )
        coreBuffer = self.theBufferFactory.createEntityPropertyListBuffer( coreFullPNList  )
        optionalBuffer = self.theBufferFactory.createEntityPropertyListBuffer( optionalFullPNList  )

        aClassName = self.theModel.getEntityClassName( self.__theID )

        # delete entity

        self.theModel.deleteEntity( self.__theID )

        # create newstepper
        self.theModel.createEntity( self.__theClassName, self.__theID )

        # paste property list
        self.theBufferPaster.pasteEntityPropertyListBuffer( coreBuffer, self.__theID, None, False )
        self.theBufferPaster.pasteEntityPropertyListBuffer( optionalBuffer, self.__theID )


        return True

    def createReverseCommand( self ):
        # store stepper
        aFullPNList = []
        for aPropertyName in self.theModel.getEntityPropertyList( self.__theID ):
            aFullPNList.append( createFullPN( self.__theID, aPropertyName ) )
        pasteBuffer = self.theBufferFactory.createEntityPropertyListBuffer( aFullPNList  )

        aClassName = self.theModel.getEntityClassName( self.__theID )

        # create chg command
        changeCommand = ChangeEntityClass( self.theReceiver, aClassName, self.__theID )

        # create paste command
        pasteCommand = PasteEntityPropertyList( self.theReceiver, self.__theID, pasteBuffer )
        
        self.theReverseCommandList = [ changeCommand ] #, pasteCommand ]


class ChangeEntityProperty(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    FULLPN = 0
    VALUE = 1

    def checkArgs( self ):
        #check if fullPN exists
        if not ModelCommand.checkArgs(self):
            return False
        self.__theFullPN = self.theArgs[ self.FULLPN ]
        if not self.isFullPNExist( self.__theFullPN ):
            return False
        # check if type is same 
        self.__theValue = self.theArgs[ self.VALUE ]
        if self.theModel.getEntityPropertyAttributes( self.__theFullPN )[MS_GETTABLE_FLAG]:
            self.__theoldProp = self.theModel.getEntityProperty(self.__theFullPN )
        else:
            self.__theoldProp = None
        aPropertyType = self.theModel.getEntityPropertyType( self.__theFullPN )
        convertedValue = DMTypeCheck( self.__theValue, aPropertyType )
        if convertedValue == None:
            return False
        else:
            self.__theValue = convertedValue
        return True

    def do( self ):
        self.theModel.setEntityProperty( self.__theFullPN, self.__theValue )
        
        
        return True

    def createReverseCommand( self ):
        self.theReverseCommandList = None
        if self.__theoldProp != None:
            revcom = ChangeEntityProperty( self.theReceiver, self.__theFullPN, self.__theoldProp )
            self.theReverseCommandList = [ revcom ]


    def getAffected( self ):
        return (getFullIDType( self.__theFullPN ), getFullID( self.__theFullPN ) )


class CreateEntityProperty(ModelCommand):
    """
    args:
    """
    FULLPN = 0
    VALUE = 1
    TYPE = 2
    ARGS_NO = 3

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        
        self.__theFullPN = self.theArgs[ self.FULLPN ]
        if not isFullIDEligible( self.__theFullPN ):
            return False

        # check if full id exists
        if not self.theModel.isEntityExist( getFullID( self.__theFullPN ) ):
            return False
        if self.isFullPNExist( self.__theFullPN ):
            return False
        self.__theValue = self.theArgs[ self.VALUE ]
        self.__theType = self.theArgs[ self.TYPE ]
        convertedValue = DMTypeCheck( self.__theValue, self.__theType )
        if convertedValue != None:
            self.__theValue = convertedValue
        else:
            return False
        
        #CHECK WHETHER PROPERTIES CAN BE CREATED WHEN DM IS AVAILABLE!!!
        aClass = self.theReceiver.getModel().getEntityClassName( getFullID( self.__theFullPN ) )
        classInfoList = self.theReceiver.getDMInfo().getClassInfoList( aClass )
        if DM_ACCEPTNEWPROPERTY in classInfoList:
            return self.theReceiver.getDMInfo().getClassInfo( aClass,  DM_ACCEPTNEWPROPERTY )
        else:
            return False
        return True

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, getFullID( self.__theFullPN ) )

    def do( self ):

        
        try:
            self.theModel.createEntityProperty( getFullID( self.__theFullPN), getPropertyName( self.__theFullPN ), self.__theValue, self.__theType )
        except Exception:
            return False
        return True
        
    def createReverseCommand( self ):
        revcom = DeleteEntityPropertyList( self.theReceiver, [ self.__theFullPN ] )
        self.theReverseCommandList = [ revcom ]


class DeleteEntityPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 1
    FULLPNLIST = 0
    # ASSUMES THAT ALL FULLPNS BELONG TO ONE FULLID!!!

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False

        self.__thePropertyList = self.theArgs[ self.FULLPNLIST ]
        # check args( at least one should be deleteable)
        for aProperty in self.__thePropertyList:
            if self.isFullPNExist( aProperty ):
                if self.theModel.getEntityPropertyAttributes( aProperty )[ME_DELETEABLE_FLAG]:
                    return True
        return False

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, getFullID( self.__thePropertyList[0] ) )

    def do( self ):
        for aFullPN in self.__thePropertyList:
            try:
                self.theModel.deleteEntityProperty( getFullID( aFullPN), getPropertyName( aFullPN ) )
            except:
                pass
        return True

    def createReverseCommand( self ):
        aFullID = getFullID( self.__thePropertyList[0] )
        propBuffer = self.theBufferFactory.createEntityPropertyListBuffer( self.__thePropertyList )
        self.theReverseCommandList = [ PasteEntityPropertyList(self.theReceiver, aFullID, propBuffer ) ]


class RenameEntityProperty(ModelCommand):
    """
    args:
    """
    ARGS_NO = 3
    FULLID = 0
    OLDNAME = 1
    NEWNAME = 2

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.FULLID ]
        if not self.theModel.isEntityExist( self.__theID ):
            return False
        self.__theOldName = self.theArgs[ self.OLDNAME]
        if self.__theOldName not in self.theModel.getEntityPropertyList( self.__theID ):
            return False
        self.__theOldFullPN = createFullPN(self.__theID, self.__theOldName)
        if not self.theModel.getEntityPropertyAttributes( self.__theOldFullPN )[ME_DELETEABLE_FLAG]:
            return False
        self.__theNewName = self.theArgs[ self.NEWNAME ]
        if not isIDEligible( self.__theNewName ):
            return False

        if self.__theNewName in self.theModel.getEntityPropertyList( self.__theID ):
            return False
        #CHECK WHETHER NEW PROPERTIES CAN BE ADDED!!!
        aClass = self.theReceiver.getModel().getEntityClassName( getFullID( self.__theOldFullPN ) )
        classInfoList = self.theReceiver.getDMInfo().getClassInfoList( aClass )
        if DM_ACCEPTNEWPROPERTY in classInfoList:
            return self.theReceiver.getDMInfo().getClassInfo( aClass,  DM_ACCEPTNEWPROPERTY )
        else:
            return False
        return True

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )

    def do( self ):
        aValue = copyValue( self.theModel.getEntityProperty(self.__theOldFullPN ) )
        aType = self.theModel.getEntityPropertyType( self.__theOldFullPN  )
        self.theModel.deleteEntityProperty( self.__theID, self.__theOldName )
        self.theModel.createEntityProperty( self.__theID, self.__theNewName, aValue, aType )
        return True

    def createReverseCommand( self ):
        reverseCommand = RenameEntityProperty( self.theReceiver, self.__theID, self.__theNewName, self.__theOldName )
        self.theReverseCommandList = [ reverseCommand ]


class CopyEntityPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO= 1
    FULLPNLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__thePropertyList = self.theArgs[ self.FULLPNLIST ]
        return True
    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createEntityPropertyListBuffer( self.__thePropertyList ) )
        return True

    def createReverseCommand( self ):
        # there is no reverse command for copy
        self.theReverseCommandList = None

    def getAffected( self ):
        return (None, None )


class CutEntityPropertyList(ModelCommand):
    """
    args:
    """

    ARGS_NO= 1
    FULLPNLIST = 0

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__thePropertyList = self.theArgs[ self.FULLPNLIST ]
        for aProperty in self.__thePropertyList:
            if self.theModel.getEntityPropertyAttributes( aProperty )[ME_DELETEABLE_FLAG]:
                return True
        return False
    
    def do( self ):
        self.theReceiver.setCopyBuffer ( self.theBufferFactory.createEntityPropertyListBuffer( self.__thePropertyList ) )
        for aFullPN in self.__thePropertyList:
            try:
                self.theModel.deleteEntityProperty( getFullID( aFullPN), getPropertyName( aFullPN ) )
            except:
                pass
        return True

    def createReverseCommand( self ):
        aFullID = getFullID( self.__thePropertyList[0] )
        propBuffer = self.theBufferFactory.createEntityPropertyListBuffer( self.__thePropertyList )
        self.theReverseCommandList = [ PasteEntityPropertyList(self.theReceiver,aFullID, propBuffer) ]

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, getFullID( self.__thePropertyList[0] ) )


class PasteEntityPropertyList(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    FULLID = 0
    BUFFER = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.FULLID ]
        if not self.theModel.isEntityExist( self.__theID ):
            return False
        self.__theBuffer = self.theArgs[ self.BUFFER ]
        if self.__theBuffer.getType() != ME_PROPERTY_TYPE:
            return False
        return True 

    def getAffected( self ):
        return (ME_PROPERTY_TYPE, self.__theID )

    def do( self ):
        fullPropertyList = self.theModel.getEntityPropertyList( self.__theID )
        bufferPropertyList = self.__theBuffer.getPropertyList()
        deleteList = []
        pasteBuffer = self.theBufferFactory.createEntityPropertyListBuffer( )
        #check if property exist
        for aProperty in bufferPropertyList:
            aFullPN = createFullPN( self.__theID, aProperty )
            if aProperty in fullPropertyList:
                if self.theModel.getEntityPropertyAttributes( aFullPN )[ME_SETTABLE_FLAG]:
                    # if exist try to change value
                    self.theBufferFactory.addToEntityPropertyListBuffer( pasteBuffer, aFullPN )
                    aValue = self.__theBuffer.getProperty(aProperty)
                    aType = self.theModel.getEntityPropertyType( aFullPN )
                    convertedValue = DMTypeCheck( aValue, aType )
                    if convertedValue != None:
                        self.theModel.setEntityProperty( aFullPN, convertedValue )
            else:
                # if not exists paste it
                deleteList.append( aFullPN )
                self.theBufferPaster.pasteEntityPropertyListBuffer( self.__theBuffer, self.__theID, aProperty )
        # create reverse command
        deleteCommand = DeleteEntityPropertyList( self.theReceiver, deleteList )
        pasteCommand = PasteEntityPropertyList( self.theReceiver, self.__theID, pasteBuffer )
        self.theReverseCommandList = [ deleteCommand, pasteCommand ]
        return True

    def createReverseCommand( self ):
        #reverse command is created while doing operation
        self.theReverseCommandList = None


class SetEntityInfo(ModelCommand):
    """
    args:
    """
    ARGS_NO = 2
    ID = 0
    STRINGS = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs(self):
            return False
        self.__theID = self.theArgs[ self.ID ]
        self.__theStrings = self.theArgs[ self.STRINGS ]
        if not self.theModel.isEntityExist( self.__theID ):
            return False
        return True

    def do( self ):
        self.theModel.setEntityInfo( self.__theID, self.__theStrings )
        
        return True

    def createReverseCommand( self ):
        aStrings =  self.theModel.getEntityInfo( self.__theID ) 
        self.theReverseCommandList = [ SetEntityInfo( self.theReceiver, self.__theID, aStrings) ]


    def getAffected( self ):
       
        return (getFullIDType( self.__theID ), self.__theID )


class RelocateEntity( ModelCommand ):
    """
    args: system cannot be entityID!!! system cannot be relocated in this version
    """
    ARGS_NO = 2
    ENTITYID = 0
    TARGETSYSTEMID = 1

    def checkArgs( self ):
        if not ModelCommand.checkArgs( self ):
            return False
        self.sourceID = self.theArgs[ self.ENTITYID ]
        self.entityType = getFullIDType( self.sourceID )
        self.theID = self.sourceID.split(':')[2]
        self.targetSystem = self.theArgs[ self.TARGETSYSTEMID ]
        self.sourceSystem = getParentSystemOfFullID( self.sourceID )
        self.targetID = ':'.join( [ self.entityType, self.targetSystem, self.theID ] )
        if self.entityType not in [ ME_VARIABLE_TYPE, ME_PROCESS_TYPE ]:
            return False
        if self.theModel.isEntityExist( self.targetID ):
            return False
        if not self.theModel.isEntityExist( self.targetID ):
            return False

        return True

    def do( self ):
        # get class
        aClass = self.theModel.getEntityClassName( self.sourceID )
        # create a buffer copy of the properties
        aFullPNList = []
        for aPropertyName in self.theModel.getEntityPropertyList( self.sourceID ):
            aFullPNList.append( createFullPN( self.sourceID, aPropertyName ) )

        propBuffer = self.theBufferFactory.createEntityPropertyListBuffer( aFullPNList  )

        if self.entityType == ME_PROCESS_TYPE:
        # for processes:
            aVarrefList = propBuffer.getProperty( MS_PROCESS_VARREFLIST )
            # modify varref properties
            for aVarref in aVarrefList:
                aVariableID = aVarref [MS_VARREF_FULLID ]
                
                if not isAbsoluteReference( aVariableID ):
                    relFlag = True
                    absoluteVariableID = getAbsoluteReference( self.sourceID, aVariableID )
                    aVarref [MS_VARREF_FULLID ] = getRelativeReference( self.targetID, absoluteVariableID )
                propBuffer.setProperty( MS_PROCESS_VARREFLIST, aVarrefList )
        else:
        # for variables:
            aProcessList = self.theModel.getEntityProperty( createFullPN( self.sourceID, ME_VARIABLE_PROCESSLIST ) )
            # modify corresponding process
            for aProcess in aProcessList:
                aVarrefList = self.theModel.getEntityProperty( createFullPN( self.aProcess, MS_PROCESS_VARREFLIST ) )
                writeFlag = False
                for aVarref in aVarrefList:
                    aVariableID = aVarref [MS_VARREF_FULLID ]
                    if not isAbsoluteReference( aVariableID ):
                        relFlag = True
                        absoluteVariableID = getAbsoluteReference( self.aProcess, aVariableID )
                    else:
                        relFlag = False
                        absoluteVariableID = aVariableID
                    if self.sourceID.lstrip(MS_VARIABLE_TYPE) == absoluteVariableID.lstrip(MS_VARIABLE_TYPE):
                        writeFlag = False
                        absoluteVariableID = self.targetID
                        if relFlag:
                            aVariableID = getRelativeReference( aProcess, absoluteVariableID )
                        else:
                            aVariableID = absoluteVariableID
                        aVarref [MS_VARREF_FULLID ] = aVariableID
                if writeFlag:
                    self.theModel.setEntityProperty( createFullPN( self.aProcess, MS_PROCESS_VARREFLIST ), aVarrefList )

        # delete entity
        self.theModel.deleteEntity( self.sourceID )

        # paste entity
        self.theModel.createEntity( aClass, self.targetID )

        # paste propertylist
        self.theBufferPaster.pasteEntityPropertyListBuffer(propBuffer, self.targetID)

    def createReverseCommand( self ):
        self.theReverseCommandList = [ RelocateEntity(self.theReceiver, self.targetID, self.sourceSystem) ]

    def getAffected( self ):
        return [ self.entityType, self.sourceSystem ]

    def getAffected2( self ):
        return [ self.entityType, self.targetSystem ]
