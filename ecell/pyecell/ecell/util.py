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

import re
import weakref
from warnings import warn

import ecell.identifiers as identifiers
from ecell.EntityStub import EntityStub
from ecell.LoggerStub import LoggerStub
from ecell.StepperStub import StepperStub
from ecell.ecs_constants import *

def createFullID( fullidstring ):
    return identifiers.FullID( fullidstring )

def createFullID_M( aType, aSystemPath, anID ):
    return identifiers.FullID( aType, aSystemPath, anID )

def createFullPN( fullpnstring ):
    return identifiers.FullPN( fullpnstring )

def createFullIDString( fullid ):
    return identifiers.FullID( fullid )

def createFullPNString( fullpn ):
    return identifiers.FullPN( fullpn )

def convertFullIDToFullPN( fullid, property='' ):
    return identifiers.FullPN( identifiers.FullID( fullid ), property )

def convertFullPNToFullID( fullpn ):
    return identifiers.FullPN( fullpn ).fullID

def validateFullID( fullid ):
    identifiers.FullID( fullid )

def validateFullPN( fullpn ):
    identifiers.FullPN( fullid )

def getPropertyName( fullpn ):
    return fullpn[ PROPERTY ]

def getSystemPath( opaque ):
    return opaque[ SYSTEMPATH ]

def convertFullIDToSystemPath( aSystemFullID ):
    return identifiers.FullID( aSystemFullID ).toSystemPath()

def convertSystemPathToFullID( aSystemPath ):
    return identifiers.SystemPath( aSystemPath ).toFullID()

def convertSystemFullID2SystemID( aSystemFullID ):
    return identifiers.FullID( aSystemFullID ).toSystemPath()

def convertSystemID2SystemFullID( aSystemPath ):
    return identifiers.SystemPath( aSystemPath ).toFullID()

def joinSystemPath( aSystemPath1, aSystemPath2 ):
    a = identifiers.SystemPath( aSystemPath1 )
    a.append( identifiers.SystemPath( aSystemPath2 ) )
    return a

def getSuperSystemPath( aSystemPath ):
    return identifiers.SystemPath( aSystemPath ).getSuperSystemPath()

def validateIDString( aString ):
    return re.match( ".*[^A-Z|_|a-z|0-9]+", aString ) == None

def validateFullIDString( aString ):
    identifiers.FullID( aString )

def convertFullIDStringToFullPNString( aFullID, aPropertyName ):
    return identifiers.FullPN( identifiers.FullID( aFullID ), aPropertyName )

def convertFullPNStringToFullIDString( aFullPN ):
    return identifiers.FullPN( aFullPN ).fullID

def getPropertyNameFromFullPNString( aFullPN ):
    return identifiers.FullPN( aFullPN ).propertyName

def convertIDListToFullIDList( aType, aParentFullID, anIDList ):
    aParentPath = identifiers.FullID( aParentFullID ).toSystemPath()
    retval = []
    for anID in anIDList:
        retval.append( identifiers.FullID( aType, aParentPath, anID ) )
    return retval

def createFullIDStringFromVariableReference( aProcessFullID, aVarref ):
    #aVarref: containing all 3 components
    aVarrefFullID = aVarref[ MS_VARREF_FULLID ]
    aVarrefFullID = getAbsoluteReference( aProcessFullID,  aVarrefFullID )
    aVarrefTuple = aVarrefFullID.split( ':' )
    aVarrefTuple[0] = 'Variable'
    return identifiers.FullID( aVarrefTuple )

def getParentSystemIDStringFromFullIDString( aFullID ):
    return identifiers.FullID( aFullID ).getSuperSystemPath()

def convertFullIDStringToSystemPath( aSystemFullID ):
    return identifiers.FullID( aSystemFullID ).toSystemPath()

def convertSystemPathToFullIDString( aSystemPath ):
    return identifiers.SystemPath( aSystemPath ).toFullID()

def getTypeFromFullIDString( aFullID ):
    return identifiers.FullID( aFullID ).getTypeName()

__deprecated__ = (
    createFullID,
    createFullID_M,
    createFullPN,
    createFullIDString,
    createFullPNString,
    convertFullIDToFullPN,
    convertFullPNToFullID,
    validateFullID,
    validateFullPN,
    getPropertyName,
    getSystemPath,
    convertFullIDToSystemPath,
    convertSystemPathToFullID,
    convertSystemFullID2SystemID,
    convertSystemID2SystemFullID,
    joinSystemPath,
    getSuperSystemPath,
    validateIDString,
    validateFullIDString,
    convertFullIDStringToFullPNString,
    convertFullPNStringToFullIDString,
    getPropertyNameFromFullPNString,
    convertIDListToFullIDList,
    createFullIDStringFromVariableReference,
    getParentSystemIDStringFromFullIDString,
    convertFullIDStringToSystemPath,
    convertSystemPathToFullIDString,
    getTypeFromFullIDString
)

__stack_depth = 0

for aFunc in __deprecated__:
    def gen( aFunc ):
        def deprecated( *args ):
            global __stack_depth
            __stack_depth += 1
            if __stack_depth > 1:
                return
            warn( 'DEPRECATED', DeprecationWarning, stacklevel = 2 )
            retval = aFunc( *args )
            __stack_depth -= 1
            return retval
        return deprecated
    globals()[ aFunc.__name__ ] = gen( aFunc )


def printProperty( s, fullpn ):
    value = s.getEntityProperty( fullpn )
    print fullpn, '\t=\t', value

def printAllProperties( s, fullid ):

    properties = s.getEntityPropertyList( fullid )
    for property in properties:
        fullpn = fullid + ':' + property
        try:
            printProperty( s, fullpn )
        except:
            print "failed to print %s:%s" % ( fullid, property )

def printStepperProperty( s, id, propertyname ):
    
    value = s.getStepperProperty( id, propertyname )
    print id, ':', propertyname, '\t=\t', value

def printAllStepperProperties( s, id ):

    properties = s.getStepperPropertyList( id )
    for property in properties:
        try:
            printStepperProperty( s, id, property )
        except:
            print "failed to print %s:%s" % ( id, property )

def copyValue ( aValue ):
    """
    in: anytype aValue
    return copy of aValue (coverts tuples to list)
    """
    if type( aValue ) == tuple or type( aValue ) == list:
        retval = []        
        for anElement in aValue:
            retval.append( copyValue( anElement ) )             
        if type( aValue ) == tuple:
            retval = tuple( retval )
        return retval
    else:
        return aValue

def guessDMTypeFromClassName( aClassName ):
    for aSuffix in [ 'Process', 'Variable', 'System', 'Stepper' ]:
        if aClassName.endswith( aSuffix ):
            return aSuffix
    return None

def getAbsoluteReference( aProcessFullID, aVariableRef ):
    if isAbsoluteVariableReference( aVariableRef ):
        return aVariableRef
    aVariable = aVariableRef.split(':')
    if aVariable[1][0] == '/':
        # absolute ref
        absolutePath = aVariable[1]
    elif aVariable[1][0] == '.':
        aProcess = aProcessFullID.split(':')[1]
        aProcessPath = aProcess.split('/')
        while True:
            if len(aProcessPath) == 0:
                break
            if aProcessPath[0] == '':
                aProcessPath.__delitem__(0)
            else:
                break
        aVariablePath = aVariable[1].split('/')
        absolutePath = ''
        while aVariablePath != []:
            pathString =  aVariablePath.pop()
            if pathString == '.':
                break
            elif pathString == '..':
                if len(aProcessPath) == 0:
                    raise Exception("BROKEN REFERENCE")
                aProcessPath.pop()
            else:
                absolutePath =  pathString + '/' + absolutePath
        oldPath = '/' + '/'.join(aProcessPath)
        absolutePath = absolutePath.rstrip('/')
        if oldPath != '/' and absolutePath != '':
            oldPath +='/'
        absolutePath =  oldPath + absolutePath

    else:
        raise Exception("INVALID REFERENCE")

    return aVariable[0] + ':' + absolutePath + ':' + aVariable[2]

def getRelativeReference( aProcessFullID, aVariableFullID ):
    return identifiers.FullID(
        aVariableFullID.typeCode,
        aVariableFullID.getSuperSystemPath().toRelative(
            aProcessFullID.getSuperSystemPath() ),
        aVariableFullID.id )

def findCommonPath( path1, path2 ):
    list1 = path1.split('/')
    list2 = path2.split('/')
    list3 = []
    while len(list1)>0 and len(list2)>0:
        if list1[0] == list2[0]:
            list3.append( list1[0] )
            list1.pop( 0 )
            list2.pop( 0 )
        else:
            break
    if len(list3) == 1:
        return '/'
    return '/'.join(list3)

def isAbsoluteVariableReference( aVariableRef ):
    aList = aVariableRef.split(':')
    return aList[1][0] == '/'

def isRelativeVariableReference( aVariableRef ):
    aList = aVariableRef.split(':')
    return aList[1][0] == '.'

def toNative( i ):
    if type( i ) == unicode:
        return str( i )
    elif type( i ) == list:
        return map( toNative, i )
    elif type( i ) == tuple:
        return tuple( map( toNative, i ) )

class WeakSet:
    def __init__( self ):
        self.data = {}

    def __contains__( self, item ):
        return weakref.ref( item ) in self.data

    def __len__( self ):
        return len( self.data )

    def __iter__( self ):
        for ref in self.data:
            yield ref()

    def add( self, item ):
        self.data[ weakref.ref( item ) ] = True

    def remove( self, item ):
        del self.data[ weakref.ref( item ) ]

    def discard( self, item ):
        try:
            del self.data[ weakref.ref( item ) ]
        except:
            pass

    def issubset( self, container ):
        if hasattr( container, '__len__' ):
            if len( container ) < len( self.data ):
                return False

        if hasattr( container, '__contains__' ):
            for ref in self.data:
                if ref() not in container:
                    return False
        else:
            count = len( self.data )
            for item in container:
                if weakref.ref( item ) in self.data:
                    count -= 1
            return count == 0
        return True

    def issuperset( self, container ):
        if hasattr( container, '__len__' ):
            if len( container ) > len( self.data ):
                return False

        for item in container:
            if weakref.ref( item ) not in self.data:
                return False
        return True

    def update( self, container ):
        for item in container:
            self.add( item )

    def union( self, container ):
        retval = WeakSet()
        retval.data = dict( self.data )
        retval.update( container )
        return retval

    def intersection( self, container ):
        retval = WeakSet()

        if hasattr( container, '__len__' ) and \
           len( container ) > len( self.data ):
            if hasattr( container, '__contains__' ):
                for ref in self.data:
                    if ref() in container:
                        retval.data[ ref ] = True
                return retval

        for item in container:
            ref = weakref.ref( item )
            if ref in self.data:
                retval.data[ ref ] = True
        return retval

    def intersection_update( self, container ):
        if hasattr( container, '__len__' ) and \
           len( container ) > len( self.data ):
            if hasattr( container, '__contains__' ):
                for ref in self.data:
                    if ref() not in container:
                        del retval.data[ ref ]
                return

        for item in container:
            ref = weakref.ref( item )
            try:
                del self.data[ ref ]
            except:
                pass

    def difference( self, container ):
        retval = WeakSet()

        if hasattr( container, '__contains__' ) and \
           ( not hasattr( container, '__len__' ) \
               or len( container ) < len( self.data ) ):
                for ref in self.data:
                    if not ref() in container:
                        retval.data[ ref ] = True
                return retval

        retval.data = dict( self.data )
        for item in container:
            try:
                del retval.data[ weakref.ref( item ) ]
            except:
                pass
        return retval

    def difference_update( self, container ):
        if hasattr( container, '__contains__' ) and \
           ( not hasattr( container, '__len__' ) \
               or len( container ) < len( self.data ) ):
            for ref in self.data:
                if ref() in container:
                    del self.data[ ref ]
            return

        for item in container:
            try:
                del self.data[ weakref.ref( item ) ]
            except:
                pass

    def symmetric_difference( self, container ):
        retval = WeakSet()
        retval.data = dict( self.data )
        for item in container:
            ref = weakref.ref( item )
            if ref in self.data:
                del retval.data[ ref ]
            else:
                retval.data[ ref ] = True
        return retval

    def symmetric_difference_update( self, container ):
        for item in container:
            ref = weakref.ref( item )
            if ref in self.data:
                del self.data[ ref ]
            else:
                self.data[ ref ] = True

    def copy( self ):
        retval = WeakSet()
        retval.data = dict( self.data )
        return retval

    def pop( self ):
        ref, dummy = self.popitem()
        return ref()

    def __le__( self, container ):
        return self.issubset( container )

    def __ge__( self, container ):
        return self.issuperset( container )

    def __or__( self, container ):
        return self.union( container )

    def __ior__( self, container ):
        self.update( container )
        return self

    def __and__( self, container ):
        return self.intersection( container )

    def __iand__( self, container ):
        self.intersect_update( container )
        return self

    def __sub__( self, container ):
        return self.difference( container )

    def __isub__( self, container ):
        self.difference_update( container )
        return self

    def __xor__( self, container ):
        return self.symmetric_difference( container )

    def __ixor__( self, container ):
        self.symmetric_difference_update( container )
        return self

    def __repr__( self ):
        return 'WeakSet([' \
            + reduce(
                lambda s, i:
                    ( s != '' and s + ', ' or '' ) + repr( i() ),
                self.data.keys(), '' ) \
            + '])'
