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

import ecell.ecs_constants as consts
import ecell.values as values
import ecell.ecs as ecs
import ecell.emc as emc
import ecell.util as util

import ecell.config as config

__all__ = (
    'DMPropertyDescriptor',
    'DMClassInfo',
    'DMInfo'
    )

DM_DEFAULTVALUES = {
    consts.DM_PROP_TYPE_REAL : 0.0,
    consts.DM_PROP_TYPE_INTEGER: 0,
    consts.DM_PROP_TYPE_STRING: "",
    consts.DM_PROP_TYPE_POLYMORPH: []
    }

TYPE_MAP = {
    consts.DM_PROP_TYPE_REAL : values.Real,
    consts.DM_PROP_TYPE_INTEGER: values.Integer,
    consts.DM_PROP_TYPE_STRING: values.String,
    consts.DM_PROP_TYPE_POLYMORPH: values.Polymorph
    }

MASTERLIST_TYPE = 0
MASTERLIST_BUILTIN = 1
MASTERLIST_INFOLIST = 2
MASTERLIST_PROPERTYLIST = 3
MASTERLIST_FLAGS = 4
MASTERLIST_NAME = 5
INFO_TYPE          = consts.CLASSINFO_TYPE
INFO_SETTABLE      = consts.CLASSINFO_SETTABLE
INFO_GETTABLE      = consts.CLASSINFO_GETTABLE
INFO_LOADABLE      = consts.CLASSINFO_LOADABLE
INFO_SAVEABLE      = consts.CLASSINFO_SAVEABLE
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

class DMPropertyDescriptor( object ):
    def __init__( self, *args, **nargs ):
        if len( args ) == 1:
            if type( args[ 0 ] ) in ( tuple, list ):
                args = args[ 0 ]
                if len( args ) != 5:
                    raise ValueError('PropertyInfo consists of 5 fields (%d given)' % len( args ) )
        elif len( args ) == 0 and len( nargs ) != 0:
            for key in nargs:
                setattr( self, key, nargs[ key ] )
            return
        elif len( args ) != 6:
            raise ValueError( 'Wrong number of arguments' \
                              + ' (%s given, 6 expected)' \
                              % len( args ) )
        self.typeCode      = args[ INFO_TYPE ]
        self.attributes    = self.convertFlagsToAttrs( args )
        self.defaultValue =  args[ INFO_DEFAULT_VALUE ]

    def convertFlagsToAttrs( aInfoList ):
        attr = 0
        if aInfoList[ INFO_SETTABLE ]:
            attr |= consts.DM_PROP_ATTR_SETTABLE
        if aInfoList[ INFO_GETTABLE ]:
            attr |= consts.DM_PROP_ATTR_GETTABLE
        if aInfoList[ INFO_LOADABLE ]:
            attr |= consts.DM_PROP_ATTR_LOADABLE
        if aInfoList[ INFO_SAVEABLE ]:
            attr |= consts.DM_PROP_ATTR_SAVEABLE
        return attr
    convertFlagsToAttrs = staticmethod( convertFlagsToAttrs )

    def __setattr__( self, aKey, aValue ):
        if aKey == 'isSettable':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_PROP_ATTR_SETTABLE ) \
                     | ( aValue and consts.DM_PROP_ATTR_SETTABLE or 0 )
        elif aKey == 'isGettable':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_PROP_ATTR_GETTABLE ) \
                     | ( aValue and consts.DM_PROP_ATTR_GETTABLE or 0 )
        elif aKey == 'isLoadable':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_PROP_ATTR_LOADABLE ) \
                     | ( aValue and consts.DM_PROP_ATTR_LOADABLE or 0 )
        elif aKey == 'isSaveable':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_PROP_ATTR_SAVEABLE ) \
                     | ( aValue and consts.DM_PROP_ATTR_SAVEABLE or 0 )
        elif aKey == 'typeCode':
            self.type = TYPE_MAP[ aValue ]
        object.__setattr__( self, aKey, aValue )

    def __getattr__( self, aKey ):
        if aKey == 'isSettable':
            return bool( self.attributes & consts.DM_PROP_ATTR_SETTABLE )
        elif aKey == 'isGettable':
            return bool( self.attributes & consts.DM_PROP_ATTR_GETTABLE )
        elif aKey == 'isLoadable':
            return bool( self.attributes & consts.DM_PROP_ATTR_LOADABLE )
        elif aKey == 'isSaveable':
            return bool( self.attributes & consts.DM_PROP_ATTR_SAVEABLE )
        elif aKey == 'isDynamic':
            return bool( self.attributes & consts.DM_PROP_ATTR_DYNAMIC )
        raise KeyError

    def __setitem__( self, anIndex, aValue ):
        if anIndex == INFO_TYPE:
            self.typeCode = aValue
        elif anIndex == INFO_SETTABLE:
            self.isSettable = aValue
        elif anIndex == INFO_GETTABLE:
            self.isGettable = aValue
        elif anIndex == INFO_LOADABLE:
            self.isLoadable = aValue
        elif anIndex == INFO_SAVEABLE:
            self.isSaveable = aValue
        elif anIndex == INFO_DEFAULT_VALUE:
            self.defaultValue = aValue
        raise KeyError

    def __getitem__( self, anIndex ):
        if anIndex == INFO_TYPE \
           or anIndex == consts.DM_PROPERTY_TYPE:
            return self.typeCode
        elif anIndex == INFO_SETTABLE \
             or anIndex == consts.DM_PROPERTY_SETTABLE_FLAG:
            return self.isSettable
        elif anIndex == INFO_GETTABLE \
             or anIndex == consts.DM_PROPERTY_GETTABLE_FLAG:
            return self.isGettable
        elif anIndex == INFO_LOADABLE \
             or anIndex == consts.DM_PROPERTY_LOADABLE_FLAG:
            return self.isLoadable
        elif anIndex == INFO_SAVEABLE \
             or anIndex == consts.DM_PROPERTY_SAVEABLE_FLAG:
            return self.isSaveable
        elif anIndex == consts.DM_PROPERTY_DELETEABLE_FLAG:
            return self.isDynamic
        elif anIndex == INFO_DEFAULT_VALUE \
             or anIndex == consts.DM_PROPERTY_DEFAULTVALUE:
            return self.defaultValue
        raise KeyError

class DMClassInfo( object ):
    def __init__( self, *args, **nargs ):
        self.frozen = False
        if len( args ) == 0:
            self.typeCode     = 0
            self.attributes   = 0
            self.description  = None
            self.baseClass    = None
            self.name         = ''
            self.propertyList = []
            if len( nargs ) > 0:
                for key in nargs:
                    setattr( self, key, nargs[ key ] )
                self.frozen = True
            return
        elif len( args ) == 1:
            if type( args[ 0 ] ) in ( tuple, list ):
                args = args[ 0 ]
                if len( args ) != 6:
                    raise ValueError('ClassInfo consists of 6 fields (%d given)' % len( args ) )
        elif len( args ) != 6:
            raise ValueError( 'Wrong number of arguments' \
                              + ' (%s given, 6 expected)' \
                              % len( args ) )
        attrs = 0
        if args[ MASTERLIST_BUILTIN ]:
            attrs |= consts.DM_ATTR_BUILTIN
        attrs |= self.convertFlagsToAttrs( args[ MASTERLIST_FLAGS ] )
        if args[ MASTERLIST_INFOLIST ].has_key( consts.DM_DESCRIPTION ):
            desc = args[ MASTERLIST_INFOLIST ][ consts.DM_DESCRIPTION ]
        else:
            desc = None
        if args[ MASTERLIST_INFOLIST ].has_key( consts.DM_BASECLASS ):
            baseClass = args[ MASTERLIST_INFOLIST ][ consts.DM_BASECLASS ]
        else:
            baseClass = None
        self.typeCode     = args[ MASTERLIST_TYPE ]
        self.attributes   = attrs
        self.description  = desc
        self.baseClass    = baseClass
        self.propertyList = args[ MASTERLIST_PROPERTYLIST ]
        self.name         = args[ MASTERLIST_NAME ]
        self.frozen = True

    def convertFlagsToAttrs( flags ):
        attrs = 0
        if type( flags ) in ( list, tuple ):
            if flags[ consts.DM_CAN_INSTANTIATE ]:
                attrs |= consts.DM_ATTR_INSTANTIABLE
            if flags[ consts.DM_CAN_LOADINFO ]:
                attrs |= consts.DM_CAN_LOADINFO
            if flags[ consts.DM_CAN_ADDPROPERTY ]:
                attrs |= consts.DM_CAN_ADDPROPERTY
        elif type( flags ) == int:
            attrs |= flags
        else:
            raise TypeError, "Invalid type for flags"
        return attrs
    convertFlagsToAttrs = staticmethod( convertFlagsToAttrs )

    def __setattr__( self, aKey, aValue ):
        if aKey != 'frozen' and self.frozen:
            raise RuntimeError, "Object is already frozen"
        if aKey == 'isBuiltin':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_ATTR_BUILTIN ) \
                     | ( aValue and consts.DM_ATTR_BUILTIN or 0 )
        elif aKey == 'flags':
            aKey = 'attributes'
            aValue = ( self.attributes & ~( consts.DM_ATTR_INSTANTIABLE
                          | consts.DM_ATTR_INFO_LOADABLE \
                          | consts.DM_ATTR_DYN_PROPERTY_SLOT ) ) \
                     | self.convertFlagsToAttrs( aValue )
        elif aKey == 'acceptsNewProperty':
            aKey = 'attributes'
            aValue = ( self.attributes & ~consts.DM_ATTR_DYN_PROPERTY_SLOT ) \
                     | ( aValue and consts.DM_ATTR_DYN_PROPERTY_SLOT or 0 )
        object.__setattr__( self, aKey, aValue )

    def __getattr__( self, aKey ):
        if aKey == 'isBuiltin':
            return bool( self.attributes & consts.DM_ATTR_BUILTIN )
        elif aKey == 'flags':
            attrs = self.attributes
            return (
                bool( attrs & consts.DM_ATTR_INSTANTIABLE ),
                bool( attrs & consts.DM_ATTR_INFO_LOADABLE ),
                bool( attrs & consts.DM_ATTR_DYN_PROPERTY_SLOT )
                )
        elif aKey == 'acceptsNewProperty':
            return bool( self.attributes & consts.DM_ATTR_DYN_PROPERTY_SLOT )
        raise KeyError

    def __setitem__( self, anIndex, aValue ):
        if anIndex == MASTERLIST_TYPE:
            self.typeCode = aValue
        elif anIndex == MASTERLIST_BUILTIN:
            self.isBuiltin = aValue
        elif anIndex == MASTERLIST_INFOLIST:
            if aValue.has_key( consts.DM_BUILTIN ):
                self.isBuiltin = aValue[ consts.DM_BUILTIN ]
            if aValue.has_key( consts.DM_ACCEPTNEWPROPERTY ):
                self.acceptsNewProperty = aValue[ consts.DM_ACCEPTNEWPROPERTY ]
        elif anIndex == MASTERLIST_PROPERTYLIST:
            self.propertyList = aValue
        elif anIndex == MASTERLIST_FLAGS:
            self.flags = aValue
        elif anIndex == consts.DM_DESCRIPTION:
            self.description = aValue
        elif anIndex == consts.DM_ACCEPTNEWPROPERTY:
            self.acceptsNewProperty = aValue
        elif anIndex == consts.DM_PROPERTYLIST:
            self.propertyList = aValue
        elif anIndex == consts.DM_BASECLASS:
            self.baseClass = aValue
        elif anIndex == consts.DM_BUILTIN:
            self.isBuiltin = aValue
        raise KeyError

    def __getitem__( self, anIndex ):
        if anIndex == MASTERLIST_TYPE:
            return self.typeCode
        elif anIndex == MASTERLIST_BUILTIN:
            return self.isBuiltin
        elif anIndex == MASTERLIST_INFOLIST:
            return self
        elif anIndex == MASTERLIST_PROPERTYLIST:
            return self.propertyList
        elif anIndex == MASTERLIST_FLAGS:
            return self.flags
        elif anIndex == consts.DM_DESCRIPTION:
            return self.description
        elif anIndex == consts.DM_ACCEPTNEWPROPERTY:
            return self.acceptsNewProperty
        elif anIndex == consts.DM_PROPERTYLIST:
            return self.propertyList
        elif anIndex == consts.DM_BASECLASS:
            return self.baseClass
        elif anIndex == consts.DM_BUILTIN:
            return self.isBuiltin
        raise KeyError

    def has_key( self, aKey ):
        return aKey in (
            consts.DM_DESCRIPTION,
            consts.DM_ACCEPTNEWPROPERTY,
            consts.DM_PROPERTYLIST,
            consts.DM_BASECLASS,
            consts.DM_BUILTIN
            )

    def freeze( self ):
        self.frozen = True

class DMInfo:
    # FIRST INITIATE HARDCODED CLASSES SUCH AS VARIABLE, SYSTEM, COMPARTMENT SYSTMEM FROM HERE
    def __init__( self ):
        self.theSimulator = emc.Simulator()
        infoList = {
            consts.DM_DESCRIPTION: "Information cannot be obtained about this class.",
            consts.DM_BASECLASS:   "Unknown",
            consts.DM_ACCEPTNEWPROPERTY : True
            }
        self.builtins = {
            'Stepper': DMClassInfo(
                consts.DM_TYPE_STEPPER,
                True,
                infoList,
                {
                    consts.DMINFO_STEPPER_PROCESSLIST: DMPropertyDescriptor(
                        consts.DM_PROP_TYPE_POLYMORPH,
                        False, True, False, False,
                        []
                        ),
                    consts.DMINFO_STEPPER_SYSTEMLIST: DMPropertyDescriptor(
                        consts.DM_PROP_TYPE_POLYMORPH,
                        False, True, False, False,
                        []
                        )
                    },
                [ False, False, False ],
                'Stepper'
                ),
            'Process': DMClassInfo(
                consts.DM_TYPE_PROCESS,
                True,
                infoList,
                {
                    consts.DMINFO_PROCESS_STEPPERID: DMPropertyDescriptor(
                        consts.DM_PROP_TYPE_STRING,
                        False, True, True, True,
                        ""
                        ),
                    consts.DMINFO_PROCESS_VARREFLIST: DMPropertyDescriptor(
                        consts.DM_PROP_TYPE_POLYMORPH,
                        True, True, True, True,
                        []
                        )
                    },
                [ False, False, False ],
                'Process'
                ),
            consts.DM_SYSTEM_CLASS: DMClassInfo(
                consts.DM_TYPE_SYSTEM,
                True,
                infoList,
                {
                    consts.DMINFO_SYSTEM_STEPPERID: DMPropertyDescriptor(
                        consts.DM_PROP_TYPE_STRING,
                        True, True, True, True,
                        ""
                        )
                    },
                [ False, False, False ],
                consts.DM_SYSTEM_CLASS
                ),
            consts.DM_VARIABLE_CLASS: DMClassInfo(
                consts.DM_TYPE_VARIABLE,
                True,
                infoList,
                {},
                [ False, False, False ],
                consts.DM_VARIABLE_CLASS
                )
            }
        if os.name != "nt":
            self.theExtension = ".so"
        else:
            self.theExtension = ".dll"
            
        # key: class name,
        # value: list [type, builtin, infolist, propertylist, flags]
        self.theMasterList = {}
        self.loadModules()

    def loadModules( self ):
        # system, variable
        self.theMasterList.clear()
        self.loadModule( consts.DM_TYPE_SYSTEM, consts.DM_SYSTEM_CLASS, True )
        self.loadModule( consts.DM_TYPE_SYSTEM, consts.DM_SYSTEM_CLASS_OLD, True )
        self.loadModule( consts.DM_TYPE_VARIABLE, consts.DM_VARIABLE_CLASS, True )
        self.scanPath( ecs.getDMSearchPath().split( os.pathsep ), False )

    def scanPath( self, aPathList, builtin ):
        aFileInfoList = []
        for aPath in aPathList:
            for aFileName in os.listdir(aPath):
                baseName, extname = os.path.splitext( aFileName )
                if extname == self.theExtension:
                    aType = util.guessDMTypeFromClassName( baseName )
                    if aType == None:
                        continue
                    aFileInfoList.append({
                        'fileName': aFileName,
                        'className': baseName,
                        'typeName': aType,
                        'flags': [ False, False, False, False ]
                        })

        self.populateBinaryProperties( aFileInfoList, aPathList )

        for aFileInfo in aFileInfoList:
            self.loadModule(
                aFileInfo['typeName'], aFileInfo['className'], 
                builtin, aFileInfo['flags'] )

    def populateBinaryProperties( self, aFileInfoList, aPathList ):
        #first instantiate
        aInfoListStr = '[' + reduce(
            lambda o, aFileInfo:
                o + '("%s", "%s"), ' % ( aFileInfo[ 'typeName' ],
                                     aFileInfo[ 'className' ] ),
            aFileInfoList, '' ) + ']'
        ( sout, sin ) = os.popen2( sys.executable )
        sout.write( MASSTESTFILE % ( aInfoListStr, os.pathsep.join( aPathList ) ) )
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
                flags[ consts.DM_CAN_LOADINFO ] = True
            if 'create' in classProperty:
                flags[ consts.DM_CAN_INSTANTIATE ] = True
            if 'prop' in classProperty:
                flags[ consts.DM_CAN_ADDPROPERTY ] = True

    def getBinaryProperties( self, aType, aName ):
        aFileInfoList = [
            {
                'typeName': aType, 'className': aName,
                'flags': [ False, False, False, False ]
                }
            ]
        self.populateBinaryProperties(
            aFileInfoList,
            ecs.getDMSearchPath() )
        return aFileInfoList[0]['flags']

    def loadModule( self, aType, aName, builtin = False, aFlags = None ):
        # loads module and fills out masterlist
        #returns False if unsuccessful
        if aFlags == None:
            aFlags = self.getBinaryProperties( aType, aName )

        if aName == consts.DM_SYSTEM_CLASS_OLD:
            aName = consts.DM_SYSTEM_CLASS

        anInfo = self.theSimulator.getClassInfo(
            aType, aName, not builtin )

        if consts.DM_ACCEPTNEWPROPERTY not in anInfo.keys():
            supposedCanAddProperty = True
        else:
            supposedCanAddProperty = anInfo[ consts.DM_ACCEPTNEWPROPERTY ]
        # overload with real value if available
        if aFlags[ consts.DM_CAN_INSTANTIATE ]:
            anInfo[ consts.DM_ACCEPTNEWPROPERTY ] = \
                aFlags[ consts.DM_CAN_ADDPROPERTY ]
        else:
            anInfo[ consts.DM_ACCEPTNEWPROPERTY ] = supposedCanAddProperty

        propertyList = {}
        
        for anInfoName in anInfo.keys():
            if anInfoName.startswith("Property__"):
                propInfo = anInfo[ anInfoName ]
                propertyList[ anInfoName[10:] ] = \
                    DMPropertyDescriptor(
                        propInfo[ consts.CLASSINFO_TYPE ],
                        propInfo[ consts.CLASSINFO_SETTABLE ],
                        propInfo[ consts.CLASSINFO_GETTABLE ],
                        propInfo[ consts.CLASSINFO_LOADABLE ],
                        propInfo[ consts.CLASSINFO_SAVEABLE ],
                        DM_DEFAULTVALUES[ propInfo[ consts.CLASSINFO_TYPE ] ]
                        )
                anInfo[
                    anInfoName
                    ]
                anInfo.__delitem__( anInfoName )
        if not consts.DM_DESCRIPTION in anInfo.keys():
            anInfo[ consts.DM_DESCRIPTION ] = "Description not provided by module author."
        if not consts.DM_BASECLASS in anInfo.keys():
            anInfo[ consts.DM_BASECLASS ] = "Unknown."

        classInfo = DMClassInfo(
            typeCode = aType,
            isBuiltin = builtin,
            propertyList = propertyList,
            baseClass = anInfo[ consts.DM_BASECLASS ],
            description = anInfo[ consts.DM_DESCRIPTION ],
            acceptsNewProperty = anInfo[ consts.DM_ACCEPTNEWPROPERTY ],
            name = aName
            )
        self.theMasterList[ aName ] = classInfo

    def getClassInfo( self, aClassName ):
        if self.theMasterList.has_key( aClassName ):
            return self.theMasterList[ aClassName ]
        aType = util.guessDMTypeFromClassName( aClassName )
        if not self.builtins.has_key( aType ):
            raise "The type for class %s is unknown!" % aClassName
        return self.builtins[ aType ]
 
    def getClassInfoList( self, type = None ):
        if type == None:
            return self.theMasterList.values()
        else:
            retval = []
            for typeInfo in self.theMasterList.itervalues():
                if typeInfo.typeCode == type:
                    retval.append( typeInfo )
            return retval

    def getClassPropertyInfo( self, aClass, aPropertyName, anInfo ):
        return self.getClassInfo( aClass ).propertyInfo[ aPropertyName ][ anInfo ]

