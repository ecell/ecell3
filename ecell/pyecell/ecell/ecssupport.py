#!/usr/bin/env python

# this module is deprecated


import string
from ECS import *

def createFullID( fullidstring ):

    aFullID = string.split( fullidstring, ':' )
    try:
        aFullID[0] = ENTITYTYPE_DICT[aFullID[0]]
    except IndexError:
        raise ValueError( "Invalid EntityTYpe string (%s)." % aFullID[0] )
    validateFullID( aFullID )
    return  tuple( aFullID )


def createFullPN( fullpnstring ):

    aFullPN = string.split( fullpnstring, ':' )
    try:
        aFullPN[0] = ENTITYTYPE_DICT[aFullPN[0]]
    except IndexError:
        raise ValueError( "Invalid EntityTYpe string (%s)." %\
                          aFullPN[0] )
    validateFullPN( aFullPN )
    return tuple( aFullPN )


def createFullIDString( fullid ):

    validateFullID( fullid )
    aTypeString = ENTITYTYPE_STRING_LIST[int(fullid[0])]
    return aTypeString + ':' + string.join( fullid[1:], ':' )


def createFullPNString( fullpn ):

    validateFullPN( fullpn )
    aTypeString = ENTITYTYPE_STRING_LIST[fullpn[0]]
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
    # root system?
    if aSystemFullID[SYSTEMPATH] == '':
        if aSystemFullID[ID] == '/':
            aNewSystemPath = '/'
    elif aSystemFullID[SYSTEMPATH] == '/':
        aNewSystemPath = aSystemFullID[SYSTEMPATH] + aSystemFullID[ID]
    else:
        aNewSystemPath = aSystemFullID[SYSTEMPATH] + '/' + aSystemFullID[ID]


    return aNewSystemPath


if __name__ == "__main__":
    
    fullid  = createFullID( 'System:/CELL/CYTOPLASM:MT0' )
    print fullid

    fullpn = createFullPN(
        'System:/CELL/CYTOPLASM:MT0:activity' )
    print fullpn

    fullidstring = createFullIDString( fullid )
    print fullidstring

    fullpnstring = createFullPNString( fullpn )
    print fullpnstring

    print convertFullIDToFullPN( fullid )

    print convertFullPNToFullID( fullpn )

    systemfullid1  = createFullID( 'System:/:CELL' )
    systemfullid2  = createFullID( 'System:/CELL:CYTOPLASM' )
    systemfullid3  = createFullID( 'System::/' )
    print createSystemPathFromFullID( systemfullid1 )
    print createSystemPathFromFullID( systemfullid2 )
    print createSystemPathFromFullID( systemfullid3 )











