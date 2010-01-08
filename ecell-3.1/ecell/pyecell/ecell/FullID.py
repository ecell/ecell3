#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

from ecs_constants import *

import types

class FullID( tuple ):
    def __init__( self , aSequence ):
        tuple.__init__( self , aSequence )
        self.validateLength()

    def validateLength( self ):
        # over-written in FullPN
        if len( self ) != 3:
            raise ValueError( 'FullID has 3 fields. ( %d given )'
                              % len( aSequence ) )

    def getEntityTYpeString( self ):
        return ENTITYTYPE_STRING_LIST[ int( self[ TYPE ] ) ]

    def getSystemPathString( self ):
        return self[ SYSTEMPATH ].getString()
    
    def getFullID( self ):
        return createFullID( self[ : 3 ] )

    def getFullPN( self , aPropertyName='' ):
        if len( self ) == 4:
            if aPropertyName == '':
                aPropertyName = self[ PROPERTY ]
        return createFullPN( tuple( self[ : 3 ] ) + ( aPropertyName , ) )

    def getString( self ):
        aTypeString = self.getEntityTYpeString()
        aSystemPathString = self.getSystemPathString()
        return aTypeString + ':' + \
               aSystemPathString + ':' + \
               ':'.join( self[ 2 : ] )

    def convertToSystemPath( self ):
        if self[ TYPE ] == SYSTEM:
            return createSystemPath( self[ SYSTEMPATH ] + ( self[ ID ] , ) )
        else:
            raise TypeError( 'can not convert FullID of %s type entity'
                             ' to SystemPath' % self.getEntityTYpeString() )

    def getSuperSystemFullID( self ):
        return self[ SYSTEMPATH ].getFullID()

class FullPN( FullID ):
    def __init__( self , aSequence ):
        FullID.__init__( self , aSequence )

    def validateLength( self ):
        if len( self ) != 4:
            raise ValueError( 'FullPN has 4 fields. ( %d given ).'
                              % len( aSequence ) )

class SystemPath( tuple ):
    def __init__( self , aSequence ):
        tuple.__init__( self , aSequence )

    def getFullID( self ):
        return createFullID( ( SYSTEM , self[ : -1 ] , self[ -1 ] ) )

    def getFullPN( self , aPropertyName = '' ):
        return self.getFullID().getFullPN( aPropertyName )

    def getString( self ):
        return '/' + '/'.join( self[ 1 : ] )

    def isAbsolute( self ):
        if self[0] == '/':
            return 1
        else :
            return 0

    def joinSystemPath( self , aRelativeSystemPath ):
        if aRelativeSystemPath.isAbsolute():
            raise ValueError( 'can not join absolute system path' )

        aSystemPath = createSystemPath( self + aRelativeSystemPath )
        aList = aSystemPath.solveParentReference()
        
        return createSystemPath( aList )

    def solveParentReference( self , aList = None ):
        if not aList:
            aList = list( self )

        if aList[0] == '..':
            aList = self.solveParentReference( aList[ 1 : ] )
            return ['..'] + aList

        try:
            aPosition = aList.index( '..' )
        except ValueError:
            return aList

        if aList[ aPosition - 1 ] == '/':
            raise ValueError( 'root system has no parent system' )
        
        aNewList = aList[ : aPosition - 1 ] + aList[ aPosition + 1 : ]
        aList = self.solveParentReference( aNewList )
        return aList
        
def createFullID( aValue ):
    aList = convertToFullIDList( aValue )
    if not aList:
        raise TypeError( 'can not create FullID from %s type object'
                         % type( aValue ) )
    else:
        aList[ SYSTEMPATH ] = createSystemPath( aList[ SYSTEMPATH ] )
        return FullID( aList )

def createFullPN( aValue ):
    aList = convertToFullIDList( aValue )
    if not aList:
        raise TypeError( 'can not create FullPN from %s type object'
                         % type( aValue ) )
    else:
        aList[ SYSTEMPATH ] = createSystemPath( aList[ SYSTEMPATH ] )
        return FullPN( aList )

def createSystemPath( aValue ):
    # empty system path
    if aValue == '':
        return SystemPath( () )

    aList = convertToSystemPathList( aValue )
    if not aList:
        raise TypeError( 'can not create SystemPath from %s type object'
                         % type( aValue ) )

    return SystemPath( aList )

def convertToList( aValue ):
    if isinstance( aValue , list ):
        return aValue
    elif isinstance( aValue , tuple ):
        return list( aValue )
    else:
        return None

def convertToFullIDList( aValue ):
    aList = convertToList( aValue )
    if aList:
        return aList

    else:
        if isinstance( aValue , str ):
            aList = aValue.split( ':' )
            try:
                aList[ 0 ] = ENTITYTYPE_DICT[ aList[ 0 ] ]
            except IndexError:
                raise ValueError( 'Invalid EntityTYpe string (%s).'
                                  % aList[ 0 ] )
            return aList

        else:
            return None

def convertToSystemPathList( aValue ):
    aList = convertToList( aValue )
    if aList:
        return aList

    else:
        if isinstance( aValue , str ):
            aList = aValue.split( '/' )
            if aValue == '':
                return []

            if aList[ 0 ] == '':
                del( aList[ 0 ])

            if aValue.lstrip()[ 0 ] == '/':
                aList[ 0 : 0 ] = ['/']

            return aList

        else:
            return None

if __name__ == '__main__':

    aSystemPath = createSystemPath( ( '/' , 'CELL' , 'CYTOPLASM' ) )
    print aSystemPath

    print

    aTuple = ( SYSTEM , ( '/' , 'CELL' , 'CYTOPLASM' ) , 'CHROMOSOME' )
    aFullID = createFullID( aTuple )
    print aFullID

    aFullID = createFullID( 'System:/CELL/CYTOPLASM:CHROMOSOME' )
    print aFullID

    aFullID = createFullID( ( SYSTEM , aSystemPath , 'CHROMOSOME' ) )
    print aFullID

    print

    aTuple = ( VARIABLE , ( '/' , 'CELL' , 'CYTOPLASM' ) , 'ATP' , 'CONC' )
    aFullPN = createFullPN( aTuple )
    print aFullPN

    aFullPN = createFullPN( 'Variable:/CELL/CYTOPLASM:ATP:CONC' )
    print aFullPN

    print

    print aSystemPath
    print aSystemPath.getFullID()
    print aSystemPath.getFullPN( 'VOLUME' )
    print aSystemPath.getString()

    print

    print aFullID
    print aFullID.getFullPN( 'VOLUME' )
    print aFullID.getFullID()
    print aFullID.getString()
    print aFullID.getSuperSystemFullID()

    try:
        print aFullID.convertToSystemPath()
    except TypeError:
        print 'type error!'


    aFullID = createFullID( 'Variable:/CELL/CYTOPLASM:ATP' )
    try:
        print aFullID.convertToSystemPath()
    except TypeError:
        print 'type error!'

    print

    print aFullPN
    print aFullPN.getFullPN( 'VOLUME' )
    print aFullPN.getFullID()
    print aFullPN.getString()

    print aFullPN.getSuperSystemFullID()

    try:
        print aFullPN.convertToSystemPath()
    except TypeError:
        print 'type error!'

    aFullPN = createFullPN( 'System:/CELL:CYTOPLASM:VOLUME' )

    try:
        print aFullPN.convertToSystemPath()
    except TypeError:
        print 'type error!'

    print 

    print aSystemPath
    print aSystemPath.isAbsolute()
    aRelativeSystemPath = createSystemPath('CHROMOSOME/GENE')
    print aRelativeSystemPath
    print aRelativeSystemPath.isAbsolute()
    print aSystemPath.joinSystemPath( aRelativeSystemPath )

    aRelativeSystemPath = createSystemPath('CHROMOSOME/../GENE')
    print aRelativeSystemPath
    print aSystemPath.joinSystemPath( aRelativeSystemPath )
    
    aRelativeSystemPath = createSystemPath('../../ENVIRONMENT')
    print aRelativeSystemPath
    print aSystemPath.joinSystemPath( aRelativeSystemPath )

    print 
    
    aSystemPath = createSystemPath( 'A/B/C' )
    aRelativeSystemPath = createSystemPath( '../E/F/../../G/H/I/J/..' )
    print aSystemPath
    print aRelativeSystemPath
    print aSystemPath.joinSystemPath( aRelativeSystemPath )
    
    print
    
    aSystemPath = createSystemPath( 'A/B/C' )
    aRelativeSystemPath = createSystemPath( '../../../E/F/../../../../G/H/I/J/..' )
    print aSystemPath
    print aRelativeSystemPath
    print aSystemPath.joinSystemPath( aRelativeSystemPath )
    
    print
    
    aSystemPath = createSystemPath( '' )
    print aSystemPath

    print
    
    print 'end'

    
