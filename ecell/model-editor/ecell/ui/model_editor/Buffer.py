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
from Constants import *
from Utils import *


class Buffer:

    """ 
    Provides a temporary storage facility similar to ModelStore, but can only store and the following types:
    StepperList
    SystemList
    ProcessList
    VariableList
    PropertyList
    (maybe it should be derived from ModelStore class)
    """

    def __init__(self):
        """ 
        the class cannot store anything else just its type on the top level
        requests for storing anything else on the top level should be requested
        """
        pass

    def getType(self):
        """
        return string Type of stored data
        """
        return self.__class__.__name__.replace('ListBuffer','')
    
    def asEml( self ):

        return "<eml></eml>"
        
    

    

class StepperListBuffer (Buffer):

    def __init__(self):

        #create data structure to store things in
        # 'self.' defines method or field, it is public by default but
        # it becomes private when beginning with double underscore '__'
        self.__theStepperList = []


    def createStepper( self, aClass, anID ):
        """create a stepper"""
        #this is a local variable (will be destroyed after this method terminates)
        aPropertyList = PropertyListBuffer()
        for aStepper in self.__theStepperList:
            if aStepper['Name'] == anID:
                raise Exception("The stepper %s already exists!"%anID)
    
        # the type of this field is dictionary which is defined by curly brackets {}
        newStepper = { 'Name' : anID, 'Class' : aClass, 'PropertyList' : aPropertyList, 'Info' : "" }

        # but this is a class member variable
        self.__theStepperList.append( newStepper )


    def getStepperList( self ):
        """
        returns list of strings
        """

        toReturn = [] #must be initialized so that the interpreter knows this is a list

        for aStepper in self.__theStepperList:
            
            toReturn.append( aStepper['Name'] )

        return toReturn


    def getStepperClassName( self, aStepperID ):
        """
        in: string aStepperID
        returns string Class if aStepperID exists
        returns None if aStepperID does not exist
        """
        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:
                
                return aStepper['Class']

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def getStepperPropertyList( self, aStepperID ):
        """
        in: string aStepperID
        returns list of propertyname strings if aStepperID exists
        returns None if aStepperID does not exist
        """

        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:

                # calls method from PropertyListBuffer
                return aStepper['PropertyList'].getPropertyList()

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def createStepperProperty( self, aStepperID, aPropertyName, aValueList, chgdFlag = 0 ):
        """
        in: string aStepperID, aPropertyName
        returns None
        """

        for aStepper in self.__theStepperList:
            if aStepper['Name'] == aStepperID:
                # calls method from PropertyListBuffer
                aStepper['PropertyList'].createProperty( aPropertyName, aValueList, chgdFlag )
                return

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def setStepperProperty( self, aStepperID, aPropertyName, aValueList, chgdFlag = 1 ):
        """
        in: string aStepperID, aPropertyName
            anyvalue aValue
        """
        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:

                # calls method from PropertyListBuffer
                aStepper['PropertyList'].setProperty( aPropertyName, aValueList, chgdFlag )
                return
        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def getStepperProperty( self, aStepperID, aPropertyName ):
        """
        in: string aStepperID, aPropertyName
        returns copy of value of property !!!!
        """

        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:

                # calls method from PropertyListBuffer
                return aStepper['PropertyList'].getProperty( aPropertyName )

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def setStepperInfo( self, aStepperID, aInfoStrings ):
        """
        in: string aStepperID, aInfoStrings
        returns list of strings aInfoStrings
        """

        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:

                # this copyValue is for copying nested list
                # it has already been developed and should be put into 
                # some util file
                aStepper['Info'] = copyValue( aInfoStrings )
                return
        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )



    def getStepperInfo( self, aStepperID ):
        """
        in: string aStepperID, aInfoStrings
        returns string infostrings
        """

        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:

                return copyValue( aStepper ['Info'] )

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


    def getPropertyBuffer( self, aStepperID ):
        for aStepper in self.__theStepperList:
            
            if aStepper['Name'] == aStepperID:
                
                return aStepper['PropertyList']

        raise Exception("Stepper %s does not exists in Buffer"%aStepperID )


class EntityListBuffer(Buffer):
    def __init__(self):
        self.theEntityList = []

    def getEntityPropertyList( self, anID ):
        """
        Returns an EntityPropertyList, None if the Entity does not exist
        """
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )

        return anEntity['PropertyList'].getPropertyList()


    def createEntityProperty( self, anID, aPropertyName, aValueList, chgdFlag = 0 ):
        """
        Create an EntityProperty
        """
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )
        anEntity['PropertyList'].createProperty(aPropertyName, aValueList, chgdFlag)


    def setEntityProperty( self, anID, aPropertyName, aValueList, chgdFlag = 1 ):
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )
        anEntity['PropertyList'].setProperty( aPropertyName, aValueList, chgdFlag  )


    def getEntityProperty( self, anID, aPropertyName ):
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )
        return anEntity['PropertyList'].getProperty( aPropertyName )        


    def setEntityInfo( self, anID, InfoStrings ):
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )
        anEntity['Info'] = copyValue( InfoStrings )


    def getEntityInfo( self, anID ):
        anEntity = self.getEntity( anID )
        if anEntity == None:
            raise Exception("%s does not exists in Buffer"%anID )
        return anEntity['Info']


        
    def createEntity( self, aClass, anID ):
        if self.getEntity( anID ) == None:
            newEntity = {'Name' : anID, 'Class': aClass, 'PropertyList':PropertyListBuffer() }
            self.theEntityList.append( newEntity )
        
        else:
            raise Exception("%s already in Buffer" % anID )


    def getEntityClassName( self, anID ):
        anEntity = self.getEntity( anID ) 
        if anEntity == None:
            raise Exception("%s not found in buffer" % anID )
        else:
            return anEntity['Class']

    
    def getEntityList( self ):
        toReturn = [] #must be initialized so that the interpreter knows this is a list
        for anEntity in self.theEntityList:
            toReturn.append( anEntity['Name'] )
        return toReturn


    def getEntity( self, anID ):
        for anEntity in self.theEntityList:
            if anEntity['Name'] == anID:
                return anEntity
        return None


    def getPropertyBuffer( self, anID ):
        anEntity = self.getEntity( anID ) 
        if anEntity == None:
            raise Exception("%s not found in buffer" % anID )
        else:
            return anEntity['PropertyList']


    def isEntityExist( self, anID ):
        if self.getEntity( anID ) == None:
    
            return True
        return False



class SystemListBuffer( EntityListBuffer ):

    ##---------------------------------------------
    ## Methods for Entity (can only be called if aType is 
    ## either "SystemList", "VariableList" or "ProcessList")
    ##---------------------------------------------


    def __init__( self, aRootDir = ''):
        self.theRootDir = aRootDir
        self.theEntityList = []


    def setRootDir( self, aRootDir ):
        self.theRootDir = aRootDir
        for aSubSystem in self.theEntityList:
            aName = aSubSystem['Name']
            newRoot = convertSysIDToSysPath( "%s:%s:%s"%(ME_SYSTEM_TYPE, self.theRootDir, aName) )
            aSubSystem['SystemList'].setRootDir(  newRoot )

    def getRootDir( self ):
        return self.theRootDir

    def createEntity( self, aClass, aFullID ):
        anEntity = aFullID.split(':')

        if self.theRootDir == '':
            if anEntity[0] != ME_SYSTEM_TYPE:
                raise Exception("First entry in systembuffer should be system entity")
            self.theRootDir = anEntity[1]

        if anEntity[0] == ME_SYSTEM_TYPE:
            if self.__getAbsoluteDir( anEntity[1] ).find( self.theRootDir ) < 0:
                raise Exception("FullID %s is not downstream of buffer root %s"%(aFullID, self.theRootDir ) )
    
        if anEntity[0] == ME_PROCESS_TYPE:
            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )
            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                aSystem['ProcessList'].createEntity( aClass, anEntity[2] )
                return

        elif anEntity[0] == ME_VARIABLE_TYPE:
            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )
            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                aSystem['VariableList'].createEntity( aClass, anEntity[2] )
                return

        # if entity is system:
        aSystem = self.getEntityOrSubSystemList( aFullID )

        if aSystem == None:
            newSystem = {'Name': anEntity[2], 'PropertyList' : PropertyListBuffer(), 'Class' : aClass,
                    'SystemList' : SystemListBuffer(convertSysIDToSysPath( self.turnFullIDAbsolute( aFullID ) ) ), 
                    'ProcessList' : ProcessListBuffer(),
                     'VariableList' : VariableListBuffer() }
            self.theEntityList.append( newSystem )
        elif type(aSystem) == type(self):
            aSystem.createEntity( aClass, self.turnFullIDAbsolute ( aFullID ) )
        else:
            raise Exception("%s already exists in buffer"%aFullID)



    def getEntityClassName( self, aFullID ):
        anEntity = aFullID.split(':')
        if anEntity[0] == ME_PROCESS_TYPE:
            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )
            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                return aSystem['ProcessList'].getEntityClassName( anEntity[2] )

        elif anEntity[0] == ME_VARIABLE_TYPE:
            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )
            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                return aSystem['VariableList'].getEntityClassName( anEntity[2] )

        aSystem = self.getEntityOrSubSystemList( aFullID )          
        if aSystem == None:
            raise Exception("Buffer does not contain %s" % aFullID )
        elif type(aSystem) == type( self ):
            return aSystem.getEntityClassName( self.turnFullIDAbsolute ( aFullID ) )
        else:
            return aSystem['Class']


    def getEntityList( self, anEntityType=None, aSystemPath=None ):
        if self.theRootDir == '':
            return []
        if anEntityType == None:
            return self.getEntityList( ME_SYSTEM_TYPE, self.theRootDir )

        if self.__getRelativeDir( aSystemPath ) == '.':
            if anEntityType == ME_SYSTEM_TYPE:
                return self.listSystems()
            else:
                raise Exception("SystemListBuffer cannot contain %s on top level" %anEntityType )

        aFullID = convertSysPathToSysID( aSystemPath )
        aSystem = self.getEntityOrSubSystemList( aFullID )
        if aSystem == None:
            raise Exception("Illegal systempath passed to buffer %s."%aSystemPath)
        elif type(aSystem) == type( self ):
            return aSystem.getEntityList( anEntityType, self.__getAbsoluteDir( aSystemPath ) )
        else:
            if anEntityType == ME_PROCESS_TYPE:
                return aSystem['ProcessList'].getEntityList()
        
            elif anEntityType == ME_SYSTEM_TYPE:
                return aSystem['SystemList'].listSystems()

            elif anEntityType == ME_VARIABLE_TYPE:
                
                return aSystem['VariableList'].getEntityList()
            else:
                raise Exception ("Wrong argument for anEntityType %s" % anEntityType)


    def listSystems(self):
        toReturn = [] #must be initialized so that the interpreter knows this is a list

        for aSystem in self.theEntityList:
            
            toReturn.append( aSystem['Name'] )

        return toReturn
        

    def getEntity( self, aFullID ):

        anEntity = aFullID.split(':')

        if anEntity[0] == ME_PROCESS_TYPE:
            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )
            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                return aSystem['ProcessList'].getEntity( anEntity[2] )

        elif anEntity[0] == ME_VARIABLE_TYPE:

            aSystem = self.getEntity( convertSysPathToSysID( anEntity[1] ) )

            if aSystem == None:
                raise Exception("Illegal fullid passed to Buffer"%aFullID )
            else:
                return aSystem['VariableList'].getEntity( anEntity[2] )

        anEntity = self.getEntityOrSubSystemList( aFullID )
        if anEntity == None:
            raise Exception("Entity %s is not in buffer." % anEntity )
        elif type(anEntity) == type( self ):
            return anEntity.getEntity( self.turnFullIDAbsolute( aFullID ) )
        else:
            return anEntity


    def getEntityOrSubSystemList( self, aFullID ):
        """
        returns reference to the list of the Entity
        or a member SystemBuffer
        returns None if aFullID not found in buffer
        """
        aSystem = aFullID.split(':')

        relDir = self.__getRelativeDir( aSystem[1] )

        if relDir == None:
            raise Exception("Illegal systempath passed to buffer %s."%aFullID)
        elif relDir == '.':
            for aSubSystem in self.theEntityList:
                if aSubSystem['Name'] == aSystem[2]:
                    return aSubSystem
            return None
        else:
            subDirs = relDir.split('/')

            for aSubSystem in self.theEntityList:
                if aSubSystem['Name'] == subDirs[1]:
                    return aSubSystem['SystemList']
            raise Exception("Illegal systempath passed to buffer %s."%aFullID)


    def turnFullIDAbsolute( self, aFullID ):
        aList = aFullID.split(':')
        if aList[1][0] == '.':
            aList[1] = self.__getAbsoluteDir( aList[1] )
        return ':'.join( aList )


    def __getRelativeDir( self, absoluteDir ):
        if (absoluteDir+'xx')[0:1] =='.':
            return absoluteDir

        if absoluteDir.find(self.theRootDir) != 0:
            return None

        if self.theRootDir != '/':
            relDir = absoluteDir.replace(self.theRootDir, ".",1)
        else:
            relDir = '.' + absoluteDir
        if relDir == './':
            relDir = '.'
        return relDir
        

    def __getAbsoluteDir( self, relativeDir ):
        if relativeDir =='':
            return '/'
        if relativeDir[0]!=".":
            if relativeDir[0] == '/':
                return relativeDir
            else:
                raise Exception("Illegal path %s"%relativeDir )
        if self.theRootDir != '/':
            absoluteDir = relativeDir.replace('.', self.theRootDir ,1)
        else:
            absoluteDir = relativeDir[1:]
        if absoluteDir == '':
            absoluteDir = '/'
        return absoluteDir



class ProcessListBuffer( EntityListBuffer ):

    ##---------------------------------------------
    ## Methods for Entity (can only be called if aType is 
    ##  "ProcessList")
    ##---------------------------------------------
    def __init__( self ):
        self.theEntityList = []


class VariableListBuffer( EntityListBuffer ):

    ##---------------------------------------------
    ## Methods for Entity (can only be called if aType is 
    ##  "VariableList" )
    ##---------------------------------------------

    def __init__( self ):
        self.theEntityList = []



class PropertyListBuffer( Buffer ):

    ##---------------------------------------------
    ## Methods for Properties (can only be called
    ## if type is "PropertyList" )
    ##---------------------------------------------
    def __init__( self ):
        self.thePropertyList = []


    def getPropertyList( self ):
        toReturn = [] #must be initialized so that the interpreter knows this is a list

        for aProperty in self.thePropertyList:
            
            toReturn.append( aProperty['Name'] )

        return toReturn
        


    def createProperty( self, aPropertyName, aValueList, changedFlag = 0 ):
        if self.__getProperty( aPropertyName ) != None:
            raise Exception("%s property already exists in Buffer"%aPropertyName )
        newProperty = { 'Name':aPropertyName, 'ValueList': copyValue( aValueList), ME_CHANGED_FLAG : changedFlag }
        self.thePropertyList.append( newProperty )


    def setProperty( self, aPropertyName, aValueList, changedFlag = 1 ):
        aProperty = self.__getProperty( aPropertyName )
        if aProperty == None:
            raise Exception("%s property does not exist in Buffer"%aPropertyName )
        aProperty['ValueList'] = copyValue( aValueList )
        aProperty[ME_CHANGED_FLAG] = changedFlag

    def getProperty( self, aPropertyName ):
        aProperty = self.__getProperty( aPropertyName )
        if aProperty == None:
            raise Exception("%s property does not exist in Buffer"%aPropertyName )
        return copyValue( aProperty['ValueList'] )
    
    def getChangedFlag( self, aPropertyName ):
        aProperty = self.__getProperty( aPropertyName )
        if aProperty == None:
            raise Exception("%s property does not exist in Buffer"%aPropertyName )
        return  aProperty[ME_CHANGED_FLAG] 
            

    def __getProperty( self, aPropertyName ):
        for aProperty in self.thePropertyList:
            if aProperty['Name'] == aPropertyName:
                return aProperty
        return None

