

import ecell._emc
from Config import *
import os
import os.path
from Constants import *
from Utils import *
import sys
import string

MASTERLIST_TYPE = 0
MASTERLIST_BUILTIN = 1
MASTERLIST_INFOLIST = 2
MASTERLIST_PROPERTYLIST = 3

DM_TRANSLATIONLIST={ "Real":DM_PROPERTY_FLOAT,
                        "Integer":DM_PROPERTY_INTEGER,
                        "String":DM_PROPERTY_STRING,
                        "Polymorph":DM_PROPERTY_NESTEDLIST}


DM_DEFAULTVALUES = { DM_PROPERTY_FLOAT : 0.0,
                        DM_PROPERTY_INTEGER: 0,
                        DM_PROPERTY_STRING:"",
                        DM_PROPERTY_NESTEDLIST: [] }

INFO_TYPE = 0
INFO_GETTABLE = 2
INFO_SETTABLE = 1
INFO_LOADABLE = 4
INFO_SAVEABLE = 3


class DMInfo:
    # FIRST INITIATE HARDCODED CLASSES SUCH AS VARIABLE, SYSTEM, COMPARTMENT SYSTMEM FROM HERE
    def __init__(self ):
        self.theSimulator = ecell._emc.Simulator()
        self.theList={}
        self.theClass=None
        self.theProcessClassList = None
        self.theStepperClassList = None
        self.theEditorClassList = None
        

        if os.uname()[0] in ["Linux", "Unix"]:
            self.theExtension = ".so"
        else:
            self.theExtension = ".dll"
            
        self.theMasterList={} #key: class name, value: list [type, builtin, descriptor]

        # load built in classes
        self.builtinPath = DM_PATH
        self.__loadBuiltins()
        self.setWorkingDir( os.getcwd() )
       

    def setWorkingDir(self, aPath):
        # changes working dir, parses working dir so and dlls
        self.workingPath = aPath
        self.refresh()

        
    def refresh( self ):
        # loads all so and dll files from DMPATH and curdir
        self.__loadBuiltins()
        self.__scanCurrentPath()
        
        
    def __deleteClasses( self, builtin ):
        # deletes classes with given conditions
        for className in self.theMasterList.keys()[:]:
            masterList = self.theMasterList[ className ]
            if  builtin == masterList[MASTERLIST_BUILTIN] :
                self.theMasterList.__delitem__( className )
                
        
        
    def __loadBuiltins( self ):
        # system, variable

        self.loadModule( ME_SYSTEM_TYPE, DM_SYSTEM_CLASS, True )
        self.loadModule( ME_SYSTEM_TYPE, DM_SYSTEM_CLASS_OLD, True )
        self.loadModule( ME_VARIABLE_TYPE, DM_VARIABLE_CLASS,True )
        self.__deleteClasses( False )
        self.__scanDMPath()
        
        
    def __scanDMPath( self ):
        # scans DMPATH for module files and parses them
        # finds .so files

        self.__scanPath( self.builtinPath )

        
    def __scanCurrentPath( self ):
        # scans current path for module files and parses them
        # finds .so files
        self.__scanPath( self.workingPath )


    def __scanPath( self, aPath ):
        # look for biinaries in Path
        builtin = aPath == self.builtinPath
        fileNames = []
        for aFileName in os.listdir(aPath):
            baseName, extname = os.path.splitext( aFileName )
            if extname == self.theExtension:
                aType = getClassTypeFromName( baseName )
                self.loadModule( aType, baseName, builtin )

   

    def loadModule( self, aType, aName, builtin = False ):
        # loads module and fills out masterlist
        #returns False if unsuccessful
        if aType == ME_SYSTEM_TYPE and aName == DM_SYSTEM_CLASS_OLD:
            anInfo = self.theSimulator.getClassInfo( aType, DM_SYSTEM_CLASS )
        else:
            anInfo = self.theSimulator.getClassInfo( aType, aName )
        propertyList = {}
        for anInfoName in anInfo.keys()[:]:
            if anInfoName.startswith("Property__"):
                propertyList[ anInfoName.replace("Property__","")] = anInfo[ anInfoName ]
                anInfo.__delitem__( anInfoName )
        if DM_DESCRIPTION not in anInfo.keys():
            anInfo[DM_DESCRIPTION] = aName
        if DM_ACCEPTNEWPROPERTY not in anInfo.keys():
            anInfo[DM_ACCEPTNEWPROPERTY] = True
        self.theMasterList[ aName ] = [ aType, builtin, anInfo, propertyList ]
        

    def __getClassInfo( self, aClassName ):
        if aClassName in self.theMasterList.keys():
            return self.theMasterList[ aClassName ][MASTERLIST_INFOLIST]
        raise Exception( "%s class doesnt exist\n",aClassName)

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
            raise Exception( "%s class doesnt exist\n", aClass)
        infoList = self.theMasterList[aClass][MASTERLIST_INFOLIST].keys()
        return infoList



    #FIVE
    def getClassInfo( self, aClass, anInfo ):
        # verify dictionary

        if not self.theMasterList.has_key(aClass):
            raise Exception( "%s class doesnt exist\n", aClass)
        anInfoList = self.theMasterList[aClass][MASTERLIST_INFOLIST]
        if anInfo not in anInfoList.keys():
            raise Exception("%s class doesnt have %s info"%(aClass, anInfo) )
        # read from dictionary
        return anInfoList[anInfo]




    def getClassPropertyInfo( self, aClass, aPropertyName, anInfo ):

        if not self.theMasterList.has_key(aClass):
            raise Exception( "%s class doesnt exist\n", aClass)
        propertyList = self.theMasterList[aClass][MASTERLIST_PROPERTYLIST]
        if aPropertyName not in propertyList.keys():
            raise Exception("%s class doesnt have %s property"%(aClass, aPropertyName) )
       
        if anInfo == DM_PROPERTY_DEFAULTVALUE:
            aType = self.__getPropertyType( aClass, aPropertyName)
            return copyValue(DM_DEFAULTVALUES[ aType ])

        elif anInfo == DM_PROPERTY_SETTABLE_FLAG:
            return self.theMasterList[aClass][MASTERLIST_PROPERTYLIST][aPropertyName][INFO_SETTABLE]


        elif anInfo == DM_PROPERTY_GETTABLE_FLAG:
            return self.theMasterList[aClass][MASTERLIST_PROPERTYLIST][aPropertyName][INFO_GETTABLE]

        elif anInfo == DM_PROPERTY_LOADABLE_FLAG:
            return self.theMasterList[aClass][MASTERLIST_PROPERTYLIST][aPropertyName][INFO_LOADABLE]

        elif anInfo == DM_PROPERTY_SAVEABLE_FLAG:
            return self.theMasterList[aClass][MASTERLIST_PROPERTYLIST][aPropertyName][INFO_SAVEABLE]

        elif anInfo == DM_PROPERTY_DELETEABLE_FLAG:
            return False

        elif anInfo == DM_PROPERTY_TYPE:
            return self.__getPropertyType(aClass,aPropertyName )


    def __getPropertyType( self, aClass, aPropertyName ):
        aType = self.theMasterList[aClass][MASTERLIST_PROPERTYLIST][aPropertyName][INFO_TYPE]
        return DM_TRANSLATIONLIST[aType]


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


