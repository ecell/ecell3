#!/usr/bin/env python

import string


# PropertyAttribute bit masks
SETABLE = 1 << 0   # == 1
GETABLE = 1 << 1   # == 2


# FullID and FullPropertyName field numbers
TYPE       = 0
SYSTEMPATH = 1
ID         = 2
PROPERTY   = 3



def parseFullID( fullidstring ):

    aFullID = tuple( string.split( fullidstring, ':' ) )
    validateFullID( aFullID )
    return  aFullID


def parseFullPropertyName( fullpropertynamestring ):

    aFullPropertyName = tuple( string.split( fullpropertynamestring, ':' ) )
    validateFullPropertyName( aFullPropertyName )
    return tuple( aFullPropertyName )


def constructFullIDString( fullid ):

    validateFullID( fullid )
    return string.join( fullid, ':' )


def constructFullPropertyNameString( fullpropertyname ):

    validateFullPropertyName( fullpropertyname )
    return string.join( fullpropertyname, ':' )


def convertToFullPropertyName( fullid, property='' ):

    validateFullID( fullid )
    # must be deep copy
    fullpropertyname = tuple( fullid ) + (property,)
    return fullpropertyname


def convertToFullID( fullpropertyname ):

    validateFullPropertyName( fullpropertyname )
    fullid = tuple( fullpropertyname[:3] )
    return fullid


def validateFullID( fullid ):

    aLength = len( fullid )
    if aLength != 3:
        raise ValueError(
            "FullID has 3 fields. ( %d given )" % aLength )


def validateFullPropertyName( fullpropertyname ):

    aLength = len( fullpropertyname )
    if aLength != 4:
        raise ValueError(
            "FullPropertyName has 4 fields. ( %d given )" % aLength )



if __name__ == "__main__":
    
    fullid  = parseFullID( 'System:/CELL/CYTOPLASM:MT0' )
    print fullid

    fullproperty =  parseFullPropertyName(
        'System:/CELL/CYTOPLASM:MT0:activity' )
    print fullproperty

    fullidstring = constructFullIDString( fullid )
    print fullidstring

    fullpropertystring = constructFullPropertyNameString( fullproperty )
    print fullpropertystring

    print convertToFullPropertyName( fullid )

    print convertToFullID( fullproperty )
