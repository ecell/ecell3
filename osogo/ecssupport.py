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

# Primitive type numbers
ENTITY     = 1
SUBSTANCE  = 2
REACTOR    = 3
SYSTEM     = 4

PrimitiveTypeString =\
( 'NONE', 'Entity', 'Substance', 'Reactor', 'System' )

PrimitiveTypeDictionary =\
{
    'Entity'   : ENTITY,
    'Substance': SUBSTANCE,
    'Reactor'  : REACTOR,
    'System'   : SYSTEM
}    


def FullID( fullidstring ):

    aFullID = string.split( fullidstring, ':' )
    try:
        aFullID[0] = PrimitiveTypeDictionary[aFullID[0]]
    except IndexError:
        raise ValueError( "Invalid PrimitiveType string (%s)." % aFullID[0] )
    validateFullID( aFullID )
    return  tuple( aFullID )


def FullPropertyName( fullpropertynamestring ):

    aFullPropertyName = string.split( fullpropertynamestring, ':' )
    try:
        aFullPropertyName[0] = PrimitiveTypeDictionary[aFullPropertyName[0]]
    except IndexError:
        raise ValueError( "Invalid PrimitiveType string (%s)." %\
                          aFullPropertyName[0] )
    validateFullPropertyName( aFullPropertyName )
    return tuple( aFullPropertyName )


def FullIDString( fullid ):

    validateFullID( fullid )
    aTypeString = PrimitiveTypeString[int(fullid[0])]
    return aTypeString + ':' + string.join( fullid[1:], ':' )


def FullPropertyNameString( fullpropertyname ):

    validateFullPropertyName( fullpropertyname )
    aTypeString = PrimitiveTypeString[fullpropertyname[0]]
    return aTypeString + ':' + string.join( fullpropertyname[1:], ':' )


def FullIDToFullPropertyName( fullid, property='' ):

    validateFullID( fullid )
    # must be deep copy
    fullpropertyname = tuple( fullid ) + (property,)
    return fullpropertyname


def FullPropertyNameToFullID( fullpropertyname ):

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


def createSystemPathFromFullID( aSystemFullID ):
    if aSystemFullID[SYSTEMPATH] == '/':
        if aSystemFullID[ID] == '/':
            aNewSystemPath = '/'
        else:
            aNewSystemPath = '/' + aSystemFullID[ID]
    else:
        aNewSystemPath = aSystemFullID[SYSTEMPATH] + '/' +\
                         aSystemFullID[ID]
    return aNewSystemPath


##########################################################################

def printProperty( sim, fullpropertyname ):
    value = sim.getProperty( fullpropertyname )
    print fullpropertyname, '\t=\t', value

def printAllProperties( sim, fullid ):
    properties = sim.getProperty( fullid +  ('PropertyList',) )
    for property in properties:
        printProperty( sim, fullid + ( property, ) )

def printList( sim, primitivetype, systempath,list ):
    for i in list:
        printAllProperties( sim, ( primitivetype, systempath, i ) )

##########################################################################


if __name__ == "__main__":
    
    fullid  = FullID( 'System:/CELL/CYTOPLASM:MT0' )
    print fullid

    fullproperty = FullPropertyName(
        'System:/CELL/CYTOPLASM:MT0:activity' )
    print fullproperty

    fullidstring = FullIDString( fullid )
    print fullidstring

    fullpropertystring = FullPropertyNameString( fullproperty )
    print fullpropertystring

    print FullIDToFullPropertyName( fullid )

    print FullPropertyNameToFullID( fullproperty )

    systemfullid1  = FullID( 'System:/:CELL' )
    systemfullid2  = FullID( 'System:/CELL:CYTOPLASM' )
    systemfullid3  = FullID( 'System:/:/' )
    print createSystemPathFromFullID( systemfullid1 )
    print createSystemPathFromFullID( systemfullid2 )
    print createSystemPathFromFullID( systemfullid3 )











