#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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

# this module is deprecated

#from ObjectStub import *
from EntityStub import *
from LoggerStub import *
from StepperStub import *

import string
from ecs_constants import *

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


def createFullIDFromSystemPath( aSystemPath ):
    if aSystemPath == '/':
        return  [ SYSTEM, '', '/' ]
        
    aPos = aSystemPath.rfind('/')
    newSysID = [SYSTEM, aSystemPath[0:aPos], aSystemPath[aPos+1:len(aSystemPath) ] ]
    if newSysID[1] == '':
        newSysID[1] = '/'
    return newSysID



def joinSystemPath( aSystemPath1, aSystemPath2 ):
    if len( aSystemPath1 ) == 0:
        return aSystemPath2

    if aSystemPath1[ -1 ] == '/':
        return aSystemPath1 + aSystemPath2
    else:
        return aSystemPath1 + '/' + aSystemPath2


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
