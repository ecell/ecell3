#!/usr/bin/env python

import string


# PropertyAttribute bit masks
SETABLE = 1 << 0   # == 1
GETABLE = 1 << 1   # == 2


# FullID and FullPN field numbers
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


def getFullID( fullidstring ):

    aFullID = string.split( fullidstring, ':' )
    try:
        aFullID[0] = PrimitiveTypeDictionary[aFullID[0]]
    except IndexError:
        raise ValueError( "Invalid PrimitiveType string (%s)." % aFullID[0] )
    validateFullID( aFullID )
    return  tuple( aFullID )


def getFullPN( fullpnstring ):

    aFullPN = string.split( fullpnstring, ':' )
    try:
        aFullPN[0] = PrimitiveTypeDictionary[aFullPN[0]]
    except IndexError:
        raise ValueError( "Invalid PrimitiveType string (%s)." %\
                          aFullPN[0] )
    validateFullPN( aFullPN )
    return tuple( aFullPN )


def getFullIDString( fullid ):

    validateFullID( fullid )
    aTypeString = PrimitiveTypeString[int(fullid[0])]
    return aTypeString + ':' + string.join( fullid[1:], ':' )


def getFullPNString( fullpn ):

    validateFullPN( fullpn )
    aTypeString = PrimitiveTypeString[fullpn[0]]
    return aTypeString + ':' + string.join( fullpn[1:], ':' )


def convertFullIDToFullPN( fullid, property='' ):

    validateFullID( fullid )
    # must be deep copy
    fullpn = tuple( fullid ) + (property,)
    return fullpn


def convertFullPNToFullID( fullpn ):

    validateFullPN( fullpn )
    fullid = tuple( fullpn[:3] )
    return fullid


def validateFullID( fullid ):

    aLength = len( fullid )
    if aLength != 3:
        raise ValueError(
            "FullID has 3 fields. ( %d given )" % aLength )


def validateFullPN( fullpn ):

    aLength = len( fullpn )
    if aLength != 4:
        raise ValueError(
            "FullPN has 4 fields. ( %d given )" % aLength )


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

def printProperty( sim, fullpn ):
    value = sim.getProperty( fullpn )
    print fullpn, '\t=\t', value

def printAllProperties( sim, fullid ):
    properties = sim.getProperty( fullid +  ('PropertyList',) )
    for property in properties:
        printProperty( sim, fullid + ( property, ) )

def printList( sim, primitivetype, systempath,list ):
    for i in list:
        printAllProperties( sim, ( primitivetype, systempath, i ) )

##########################################################################


if __name__ == "__main__":
    
    fullid  = getFullID( 'System:/CELL/CYTOPLASM:MT0' )
    print fullid

    fullpn = getFullPN(
        'System:/CELL/CYTOPLASM:MT0:activity' )
    print fullpn

    fullidstring = getFullIDString( fullid )
    print fullidstring

    fullpnstring = getFullPNString( fullpn )
    print fullpnstring

    print convertFullIDToFullPN( fullid )

    print convertFullPNToFullID( fullpn )

    systemfullid1  = getFullID( 'System:/:CELL' )
    systemfullid2  = getFullID( 'System:/CELL:CYTOPLASM' )
    systemfullid3  = getFullID( 'System:/:/' )
    print createSystemPathFromFullID( systemfullid1 )
    print createSystemPathFromFullID( systemfullid2 )
    print createSystemPathFromFullID( systemfullid3 )











