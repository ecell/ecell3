import re
from decimal import Decimal

import ecell.identifiers as identifiers
import ecell.ecs_constants as consts

__all__ = (
    'Real',
    'Integer',
    'String',
    'PolymorphVector',
    'Polymorph',
    'VariableReference',
    'VARREF_NAME', 'VARREF_FULLID', 'VARREF_COEF'
    )

VARREF_NAME = 0
VARREF_FULLID = 1
VARREF_COEF = 2

class Value( object ):
    pass

class Real( Value ):
    typeCode = consts.DM_PROP_TYPE_REAL

    def convertToPythonType( self, val ):
        return float( val )
    convertToPythonType = classmethod( convertToPythonType )

    def convertToNative( self, val ):
        return float( val )
    convertToNative = classmethod( convertToNative )

class Integer( Value ):
    typeCode = consts.DM_PROP_TYPE_INTEGER

    def convertToPythonType( self, val ):
        return int( val )
    convertToPythonType = classmethod( convertToPythonType )

    def convertToNative( self, val ):
        return int( val )
    convertToNative = classmethod( convertToNative )

class String( Value ):
    typeCode = consts.DM_PROP_TYPE_STRING

    def convertToPythonType( self, val ):
        return str( val )
    convertToPythonType = classmethod( convertToPythonType )

    def convertToNative( self, val ):
        return str( val )
    convertToNative = classmethod( convertToNative )

class PolymorphVector( Value ):
    typeCode = consts.DM_PROP_TYPE_POLYMORPH

    def convertToPythonType( self, val ):
        return list( val )
    convertToPythonType = classmethod( convertToPythonType )

    def convertToNative( self, val ):
        return list( val )
    convertToNative = classmethod( convertToNative )

class Polymorph( Value ):
    typeCode = consts.DM_PROP_TYPE_POLYMORPH

    def Polymorph( self, aValue, nativeType = None ):
        if nativeType == None:
            valueType = type( aValue )
            if valueType in ( tuple, list ):
                nativeType = PolymorphVector
            elif valueType in ( str, unicode ):
                nativeType = String
            elif valueType in ( int, long ):
                nativeType = Integer
            elif valueType in ( float, Decimal ):
                nativeType = Real
            else:
                raise ValueError( 'Cannot guess the native type from the value of type %s' % valueType )
        self.nativeType = nativeType
        self.theValue = aValue

    def toNative( self ):
        return self.nativeType.convertToNative( self.theValue )

    def convertToPythonType( self, val ):
        return val
    convertToPythonType = classmethod( convertToPythonType )

    def convertToNative( self, val ):
        if val.__type__ in ( VariableReference, identifiers.FullID, identifiers.FullPN ):
            return list( val )
        elif isinstance( val, identifiers.SystemPath ):
            return str( val )
        return val
    convertToNative = classmethod( convertToNative )

class VariableReference( object ):
    def __init__( self, *args ):
        if len( args ) == 1:
            if type( args[ 0 ] ) in ( tuple, list ):
                args = args[ 0 ]
                if len( args ) != 3:
                    raise ValueError( 'Variable reference consists of 3 fields. (%d given)' % len( args ) )
            else:
                raise TypeError
        elif len( args ) != 3:
            raise ValueError( 'Wrong number of arguments' \
                              + ' (%s given, 3 expected)' \
                              % len( args ) )
        self.name        = args[ VARREF_NAME ]
        self.fullID      = identifiers.FullID( args[ VARREF_FULLID ] )
        self.coefficient = args[ VARREF_COEF ]

    def __setattr__( self, aKey, aValue ):
        if aKey == 'id':
            if re.match( "[^A-Z_a-z0-9]", aValue ) != None:
                raise ValueError( "Invalid ID string (%s).", aValue )
        elif aKey == 'coefficient' and type( aValue ) in ( str, unicode ):
            if re.match( "[^0-9Ee.+-]", aValue ) != None:
                raise ValueError( "Invalid coefficient (%s).", aValue )
        object.__setattr__( self, aKey, aValue )

    def __getitem__( self, anIndex ):
        if anIndex == VARREF_NAME:
            return self.name
        elif anIndex == VARREF_FULLID:
            return self.fullID
        elif anIndex == VARREF_COEF:
            return self.coefficient
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __setitem__( self, anIndex, aValue ):
        if anIndex == VARREF_NAME:
            self.name = aValue
        elif anIndex == VARREF_FULLID:
            self.fullID = aValue
        elif anIndex == VARREF_COEF:
            self.coefficient = aValue
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __iter__( self ):
        i = 0
        while i < 3:
            yield self[ i ]
            i += 1
