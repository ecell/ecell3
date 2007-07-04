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


import os
import sys

import ecell.ecs
import ecell.emc
from Constants import *
from Utils import *
import config

DM_TRANSLATIONLIST={ "Real":DM_PROPERTY_FLOAT,
                        "Integer":DM_PROPERTY_INTEGER,
                        "String":DM_PROPERTY_STRING,
                        "Polymorph":DM_PROPERTY_NESTEDLIST}


DM_DEFAULTVALUES = { DM_PROPERTY_FLOAT : 0.0,
                        DM_PROPERTY_INTEGER: 0,
                        DM_PROPERTY_STRING:"",
                        DM_PROPERTY_NESTEDLIST: [] }

INFO_TYPE = 0
INFO_SETTABLE = 1
INFO_GETTABLE = 2
INFO_LOADABLE = 3
INFO_SAVEABLE = 4
INFO_DEFAULT_VALUE = 5

MASSTESTFILE = '''
from ecell.ecs import *
from ecell.emc import *
import sys
aInfoList = %s
setDMSearchPath('%s')

aSimulator = Simulator()
for aType, aClass in aInfoList:
    diag = []
    try:
        aSimulator.getClassInfo( aType, aClass )
        diag.append( 'info' )
    except:
        pass
    if aType == "Stepper":
        try:
            aSimulator.createStepper( aClass, aClass )
            diag.append( 'create' )
            aSimulator.setStepperProperty( aClass, "uu", 1 )
            diag.append( 'prop' )
        except:
            pass
    else:
        fpn = aType + ":/:" + aClass
        try:
            aSimulator.createEntity( aClass, fpn )
            diag.append( 'create' )
            aSimulator.setEntityProperty( fpn+":uu", 1 )
            diag.append( 'prop' )
        except:
            pass
    print aClass + "\t" + "\t".join( diag )
'''
    

class DMInfo:
    # FIRST INITIATE HARDCODED CLASSES SUCH AS VARIABLE, SYSTEM, COMPARTMENT SYSTMEM FROM HERE
    def __init__( self ):
        self.theSimulator = ecell.emc.Simulator()
        self.__dummiesList = {}
        self.__createDummies()
        if os.name != "nt":
            self.theExtension = ".so"
        else:
            self.theExtension = ".dll"
            
        # key: class name,
        # value: list [type, builtin, infolist, propertylist, flags]
        self.theMasterList={}

        # load built in classes
        self.setWorkingDir( os.getcwd() )
        self.refresh()

    def __createDummies( self ):
        # createdummy stepper
        infoList = {
            DM_DESCRIPTION: "Information cannot be obtained about this class.",
            DM_BASECLASS:   "Unknown",
            DM_ACCEPTNEWPROPERTY : True
            }
        flags = [ False, False, False ]

        #stepper
        stepperPropertyList = {
            ME_STEPPER_PROCESSLIST: [
                "Polymorph",
                0, 1, 0, 0,
                []
                ],
            ME_STEPPER_SYSTEMLIST: [
                "Polymorph",
                0, 1, 0, 0,
                []
                ] 
            }
        stepperInfo = dict( infoList )
        stepperInfo[DM_PROPERTYLIST] = stepperPropertyList.keys()
        
        self.__dummiesList[ME_STEPPER_TYPE] = [
            ME_STEPPER_TYPE,
            True,
            stepperInfo,
            stepperPropertyList,
            flags
            ]

        #process
        processPropertyList = {
            ME_STEPPERID: [
                "String",
                1, 1, 1, 1,
                ""
                ],
            ME_PROCESS_VARREFLIST: [
                 "Polymorph",
                1, 1, 1, 1,
                []
                ]
            }
        processInfo = dict(infoList )
        processInfo[ DM_PROPERTYLIST ] = processPropertyList.keys()
        
        self.__dummiesList[ ME_PROCESS_TYPE ] = [
            ME_PROCESS_TYPE,
            True,
            processInfo,
            processPropertyList,
            flags
            ]
        #system
        systemPropertyList = {
            ME_STEPPERID : [
                "String",
                1, 1, 1, 1,
                ""
                ]
            }
        systemInfo = dict( infoList )
        systemInfo[ DM_PROPERTYLIST ] = systemPropertyList.keys()
        
        self.__dummiesList[ ME_SYSTEM_TYPE ] = [
            ME_SYSTEM_TYPE,
            True,
            systemInfo,
            systemPropertyList,
            flags
            ]
        #variable
        variablePropertyList = {
            ME_VARIABLE_PROCESSLIST: [
                "Polymorph",
                0, 1, 0, 0,
                []
                ]
            }
        variableInfo = dict( infoList )
        variableInfo[ DM_PROPERTYLIST ] = variablePropertyList.keys()
        
        self.__dummiesList[ ME_VARIABLE_TYPE ] = [
            ME_VARIABLE_TYPE,
            True,
            variableInfo,
            variablePropertyList,
            flags
            ]

    def setWorkingDir( self, aDir ):
        # changes working dir, parses working dir so and dlls
        self.workingDir = aDir
        self.refresh()

    def refresh( self ):
        # loads all so and dll files from DMPATH and curdir
        self.__loadModules()

    def __deleteClasses( self, builtin ):
        # deletes classes with given conditions
        for className in self.theMasterList.keys():
            masterList = self.theMasterList[ className ]
            if builtin == masterList[ MASTERLIST_BUILTIN ]:
                del self.theMasterList[ className ]

    def __loadModules( self ):
        # system, variable
        self.loadModule( ME_SYSTEM_TYPE, DM_SYSTEM_CLASS, True )
        self.loadModule( ME_SYSTEM_TYPE, DM_SYSTEM_CLASS_OLD, True )
        self.loadModule( ME_VARIABLE_TYPE, DM_VARIABLE_CLASS, True )
        variableDesc = self.__getClassDescriptor( DM_VARIABLE_CLASS )
        variableDesc[MASTERLIST_PROPERTYLIST][MS_VARIABLE_PROCESSLIST] = [
            "Polymorph",
            0, 1, 0, 0
            ]
        newList = list(variableDesc[MASTERLIST_INFOLIST][DM_PROPERTYLIST])
        newList.append( MS_VARIABLE_PROCESSLIST )
        variableDesc[MASTERLIST_INFOLIST][DM_PROPERTYLIST] = newList

        self.__deleteClasses( False )
        self.__scanPath( config.dm_path, False )
        self.__scanPath( [ self.workingDir ] , False )

    def __scanPath( self, aPathList, builtin ):
        aFileInfoList = []
        for aPath in aPathList:
            for aFileName in os.listdir(aPath):
                baseName, extname = os.path.splitext( aFileName )
                if extname == self.theExtension:
                    aType = getClassTypeFromName( baseName )
                    if aType == None:
                        continue
                    aFileInfoList.append({
                        'fileName': aFileName,
                        'className': baseName,
                        'typeName': aType,
                        'flags': [ False, False, False, False ]
                        })

        self.populateBinaryProperties( aFileInfoList, os.sep.join( aPathList ) )

        for aFileInfo in aFileInfoList:
            self.loadModule(
                aFileInfo['typeName'], aFileInfo['className'], 
                builtin, aFileInfo['flags'] )

    def populateBinaryProperties( self, aFileInfoList, aDMPath ):
        #first instantiate
        aInfoListStr = '[' + reduce(
            lambda o, aFileInfo:
                o + '("%s", "%s"), ' % ( aFileInfo[ 'typeName' ],
                                     aFileInfo[ 'className' ] ),
            aFileInfoList, '' ) + ']'
        ( sout, sin ) = os.popen2( sys.executable )
        sout.write( MASSTESTFILE % ( aInfoListStr, aDMPath ) )
        sout.close()
        result = sin.readlines()
        sin.close()
        classPropertyList = map(
            lambda s: s.strip( "\n" ).split( "\t" ),
            result )
        for idx in range( 0, len( classPropertyList ) ):
            classProperty = classPropertyList[ idx ]
            classProperty.pop( 0 ) # class name
            flags = aFileInfoList[ idx ][ 'flags' ]
            if 'info' in classProperty:
                flags[DM_CAN_LOADINFO] = True
            if 'create' in classProperty:
                flags[DM_CAN_INSTANTIATE] = True
            if 'prop' in classProperty:
                flags[DM_CAN_ADDPROPERTY] = True

    def __getBinaryProperties( self, aType, aName ):
        aFileInfoList = [
            {
                'typeName': aType, 'className': aName,
                'flags': [ False, False, False, False ]
                }
            ]
        self.populateBinaryProperties(
            aFileInfoList,
            ecell.ecs.getDMSearchPath() )
        return aFileInfoList[0]['flags']

    def loadModule( self, aType, aName, builtin = False, aFlags = None ):
        # loads module and fills out masterlist
        #returns False if unsuccessful
        if aFlags == None:
            aFlags = self.__getBinaryProperties( aType, aName )
        if not  aFlags[DM_CAN_LOADINFO] :
            # not info can be obtained valid DM file
            return
        else:
            if aType == ME_SYSTEM_TYPE and aName == DM_SYSTEM_CLASS_OLD:
                anInfo = self.theSimulator.getClassInfo(
                    aType, DM_SYSTEM_CLASS )
            else:
                anInfo = self.theSimulator.getClassInfo(
                    aType, aName, not builtin )
            if DM_ACCEPTNEWPROPERTY not in anInfo.keys():
                supposedCanAddProperty = True
            else:
                supposedCanAddProperty = anInfo[DM_ACCEPTNEWPROPERTY]
            # overload with real value if available
            if aFlags[DM_CAN_INSTANTIATE]:
                anInfo[DM_ACCEPTNEWPROPERTY] = aFlags[DM_CAN_ADDPROPERTY]
            else:
                anInfo[DM_ACCEPTNEWPROPERTY] = supposedCanAddProperty

        propertyList = {}
        
        for anInfoName in anInfo.keys()[:]:
            if anInfoName.startswith("Property__"):
                propertyList[ anInfoName.replace("Property__","")] = anInfo[
                    anInfoName
                    ]
                anInfo.__delitem__( anInfoName )
        if DM_DESCRIPTION not in anInfo.keys():
            anInfo[DM_DESCRIPTION] = "Description not provided by module author."
        if DM_BASECLASS not in anInfo.keys():
            anInfo[DM_BASECLASS] = "Unknown."
            
        self.theMasterList[ aName ] = [ aType, builtin, anInfo, propertyList ]

    def __getClassDescriptor( self, aClassName ):
        if aClassName in self.theMasterList.keys():
            return self.theMasterList[ aClassName ]
        aType = getClassTypeFromName( aClassName )
        if aType not in self.__dummiesList.keys():
            raise "The type of class %s is unknown!"%aClassName
        return self.__dummiesList[ aType ]
        
    # SECOND DO THIS
    def getClassList( self, aType ):
        classNames = []
        for aClassName in self.theMasterList.keys():
            if self.theMasterList[ aClassName ][MASTERLIST_TYPE] == aType:
                classNames.append( aClassName )
        return classNames

    #THIRD
    def getClassInfoList( self, aClass ):
        if not self.theMasterList.has_key(aClass):
            # return basic properties of dummy class
            return [
                DM_ACCEPTNEWPROPERTY,
                DM_PROPERTYLIST,
                DM_DESCRIPTION,
                DM_BASECLASS,
                DM_BUILTIN
                ]
            #raise Exception( "%s class doesnt exist\n", aClass)
        infoList = self.theMasterList[aClass][MASTERLIST_INFOLIST].keys()
        return infoList

    #FIVE
    def getClassInfo( self, aClass, anInfo ):
        # verify dictionary
        descriptor = self.__getClassDescriptor( aClass )
        anInfoList = descriptor[ MASTERLIST_INFOLIST ]
        if anInfo == DM_BUILTIN:
            return descriptor[ MASTERLIST_BUILTIN ]
        if anInfo not in anInfoList.keys():
            raise Exception("%s class doesnt have %s info"%(aClass, anInfo) )
        # read from dictionary
        
        return anInfoList[anInfo]

    def getClassPropertyInfo( self, aClass, aPropertyName, anInfo ):
        propertyList = self.__getPropertyInfo( aClass, aPropertyName )
        return {
            DM_PROPERTY_DEFAULTVALUE:
                lambda: copyValue(DM_DEFAULTVALUES[
                    self.__getPropertyType( aClass, aPropertyName) ]),
            DM_PROPERTY_SETTABLE_FLAG:
                lambda: propertyList[aPropertyName][INFO_SETTABLE],
            DM_PROPERTY_GETTABLE_FLAG:
                lambda: propertyList[aPropertyName][INFO_GETTABLE],
            DM_PROPERTY_LOADABLE_FLAG:
                lambda: propertyList[aPropertyName][INFO_LOADABLE],
            DM_PROPERTY_SAVEABLE_FLAG:
                lambda: propertyList[aPropertyName][INFO_SAVEABLE],
            DM_PROPERTY_DELETEABLE_FLAG:
                lambda: False, 
            DM_PROPERTY_TYPE:
                lambda: self.__getPropertyType( aClass, aPropertyName )
            }[anInfo]()

    def __getPropertyType( self, aClass, aPropertyName ):
        propertyList = self.__getPropertyInfo(aClass, aPropertyName)
        aType = propertyList[aPropertyName][INFO_TYPE]
        return DM_TRANSLATIONLIST[aType]

    def __getPropertyInfo( self, aClass, aPropertyName ):
        propertyList = self.__getClassDescriptor(aClass)[MASTERLIST_PROPERTYLIST]
        if aPropertyName not in propertyList:
            if self.__getClassDescriptor(aClass)[MASTERLIST_INFOLIST][DM_ACCEPTNEWPROPERTY] == True:
                propertyList = dict( propertyList )
                propertyList[aPropertyName] = ["String", 1,1,1,1] 
            else:
                raise("aClass %s propertyname %s does not exist and class doesnot accept new properties!"%(aClass,aPropertyName))
        return propertyList

def DMTypeCheck( aValue, aType ):
    if aType == DM_PROPERTY_STRING:
        if type(aValue) == type([]):
            aValue = aValue[0]
        return str( aValue )
    elif aType == DM_PROPERTY_MULTILINE:
        if type(aValue) == type([]):
            aValue = '/n'.join(aValue)
        return str( aValue )
    elif aType == DM_PROPERTY_NESTEDLIST:
        if type(aValue) in ( type( () ), type( [] ) ):
            return aValue
        else:
            return None
    elif aType == DM_PROPERTY_INTEGER:
        if type(aValue) == type([]):
            aValue = aValue[0]
        try:
            aValue = int( aValue )
        except:
            return None
        
        return aValue
    elif aType == DM_PROPERTY_FLOAT:
        if type(aValue) == type([]):
            aValue = aValue[0]
        try:
            aValue = float( aValue )
        except:
            return None
        return aValue
    else:
        return None
