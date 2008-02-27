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

import sys
import re

import ecell.ecs_constants as consts

__all__ = (
    'FullID',
    'FullPN',
    'SystemPath',
    'ROOT_SYSTEM_FULLID'
    )

class FullID( object ):
    def __init__( self, *args ):
        if len( args ) == 1:
            aSeq = args[0]
        elif len( args ) == 3:
            aSeq = args
        else:
            raise ValueError( 'Wrong number of arguments' \
                              + ' (%s given, 1 or 3 expected)' \
                              % len( args ) )

        object.__setattr__( self, 'image', None )
        if type( aSeq ) in ( str, unicode ):
            aSeq = aSeq.split(':')
        if type( aSeq ) in ( list, tuple ):
            if len( aSeq ) != 3:
                raise ValueError( 'FullID consists of 3 fields. ( %d given )'
                                  % len( aSeq ) )
            self.typeCode         = aSeq[ consts.TYPE ]
            self.systemPathString = str( aSeq[ consts.SYSTEMPATH ] )
            if aSeq[ consts.ID ] == '':
                if self.systemPathString != '/':
                    raise ValueError( 'ID is empty' )
                self.systemPathString = ''
                self.id = '/'
            else:
                self.id = aSeq[ consts.ID ]
        elif isinstance( aSeq, FullID ):
            self.typeCode         = aSeq.typeCode
            self.systemPathString = aSeq.systemPathString
            self.id               = aSeq.id
        else:
            raise ValueError( 'Cannot construct FullID from a value ' \
                              + 'typed %s', aSeq.__class__ )

    def getTypeName( self ):
        return consts.ENTITYTYPE_STRING_LIST[ self.typeCode ]

    def getSuperSystemPath( self ):
        if self.systemPathString == '':
            return None
        return SystemPath( self.systemPathString )

    def createFullPN( self, aPropertyName ):
        return FullPN( self, aPropertyName )

    def toSystemPath( self ):
        if self.typeCode != consts.SYSTEM:
            raise TypeError( 'can not convert FullID of %s type entity'
                             ' to SystemPath' % self.getTypeName() )
        if self.systemPathString == '' and self.id == '/':
            # somewhat an irregular case; should spew a warning if possible.
            aSysPath = SystemPath( '/' )
        else:
            aSysPath = SystemPath( self.systemPathString )
            aSysPath.append( self.id )
        return aSysPath

    def toTuple( self ):
        return ( self.typeCode, self.systemPathString, self.id )

    def __str__( self ):
        if self.image == None:
            object.__setattr__( self, 'image',
                '%s:%s:%s' % (
                    self.getTypeName(),
                    self.systemPathString,
                    self.id ) )
        return self.image

    def __setattr__( self, aKey, aValue ):
        if aKey == 'typeCode':
            if type( aValue ) in ( str, unicode ):
                try:
                    aValue = consts.ENTITYTYPE_DICT[ aValue ]
                except:
                    raise ValueError( "Invalid entity type string `%s'." \
                                      % aValue )
        elif aKey == 'id':
            if self.typeCode == consts.SYSTEM:
                if aValue != '/' and \
                   re.match( "[^A-Z_a-z0-9]",  aValue ) != None:
                    raise ValueError( "Invalid ID string (%s).", aValue )
            else:
                if re.match( "[^A-Z_a-z0-9]",  aValue ) != None:
                    raise ValueError( "Invalid ID string (%s).", aValue )
        object.__setattr__( self, aKey, aValue )
        object.__setattr__( self, 'image', None )

    def __getitem__( self, anIndex ):
        if anIndex == 0:
            return self.typeCode
        elif anIndex == 1:
            return self.systemPathString
        elif anIndex == 2:
            return self.id
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __setitem__( self, anIndex, aValue ):
        if anIndex == 0:
            self.typeCode = aValue
        elif anIndex == 1:
            self.systemPathString = aValue
        elif anIndex == 2:
            self.id = aValue
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __len__( self ):
        return 3

    def __iter__( self ):
        anIndex = 0
        while anIndex < 3:
            yield self[ anIndex ]
            anIndex += 1

    def __hash__( self ):
        return ( ( self.typeCode + hash( self.systemPathString ) << 4 ) ^ hash( self.id ) ) & sys.maxint

    def __eq__( self, theOther ):
        if not isinstance( theOther, FullID ):
            return False
        return self.typeCode == theOther.typeCode and \
               self.systemPathString == theOther.systemPathString and \
               self.id == theOther.id

class FullPN( object ):
    def __init__( self, *args ):
        if len( args ) == 1:
            if type( args[ 0 ] ) in ( list, tuple ):
                aSeq = args[ 0 ]
                aFullID = FullID( aSeq[ 0: 3 ] )
                aPropertyName = aSeq[ 3 ]
            elif type( args[ 0 ] ) in ( str, unicode ):
                aSeq = args[ 0 ].split( ':' )
                aFullID = FullID( aSeq[ 0: 3 ] )
                aPropertyName = aSeq[ 3 ]
            elif isinstance( args[ 0 ], FullPN ):
                aFullID = FullID( args[ 0 ].fullID )
                aPropertyName = args[ 0 ].propertyName
            else:
                raise ValueError( "Argument #1 should be a list or tuple" )
        elif len( args ) == 2:
            if not isinstance( args[ 0 ], FullID ):
                raise ValueError( "Argument #1 should a FullID instance" )
            aFullID = args[ 0 ]
            aPropertyName = args[ 1 ]
        elif len( args ) == 4:
            aFullID = FullID( args[ 0: 3 ] )
            aPropertyName = args[ 3 ]
        else:
            raise ValueError( 'Wrong number of arguments' \
                              + ' (%s given, 1 or 3 expected)' \
                              % len( args ) )
        object.__setattr__( self, 'image', None )
        self.fullID = aFullID
        self.propertyName = aPropertyName

    def __str__( self ):
        if self.image == None:
            object.__setattr__( self, 'image',
                '%s:%s:%s:%s' % (
                    self.fullID.getTypeName(),
                    self.fullID.systemPathString,
                    self.fullID.id,
                    self.propertyName ) )
        return self.image

    def __setattr__( self, aKey, aValue ):
        if aKey == 'propertyName':
            if re.match( "[^A-Z_a-z0-9]",  aValue ) != None:
                raise ValueError( "Invalid ID string (%s).", aValue )
            object.__setattr__( self, aKey, aValue )
        if aKey == 'fullID':
            assert isinstance( aValue, FullID )
            object.__setattr__( self, aKey, aValue )
        else:
            setattr( self.fullID, aKey, aValue )
        object.__setattr__( self, 'image', None )

    def __getitem__( self, anIndex ):
        if anIndex < 3:
            return self.fullID[ anIndex ]
        elif anIndex == 3:
            return self.propertyName
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __setitem__( self, anIndex, aValue ):
        if anIndex < 3:
            self.fullID[ anIndex ] = aValue
        elif anIndex == 3:
            self.propertyName = aValue
        else:
            raise IndexError( 'Index out of range (%d)' % anIndex )

    def __len__( self ):
        return 4

    def __iter__( self ):
        anIndex = 0
        while anIndex < 4:
            yield self[ anIndex ]
            anIndex += 1

    def __hash__( self ):
        return hash( self.fullID ) + hash( self.propertyName )

    def __eq__( self, theOther ):
        if not isinstance( theOther, FullPN ):
            return False
        return self.fullID == theOther.fullID and \
               self.propertyName == theOther.propertyName

class SystemPath:
    def __init__( self, aPath, *args ):
        assert aPath != None
        if type( aPath ) in ( list, tuple ):
            self.image = None
            self.seq = list( aPath )
            if len( self.seq ) == 0:
                self.seq.append( '.' )
        elif type( aPath ) in ( str, unicode ):
            if aPath == '':
                aPath = '.'
            else:
                aPath = aPath.rstrip( '/' )
                if aPath == '':
                    aPath = '/'
            self.image = aPath
            self.seq = None
        elif isinstance( aPath, SystemPath ):
            self.image = aPath.image
            self.seq = None
            if aPath.seq != None:
                self.seq = list( aPath.seq )
        else:
            raise ValueError( 'Invalid type for the first argument' )

        self.isCanonicalized = False

        for aPathRepr in args:
            self.append( aPathRepr )

    def ensureImageIsAvailable( self ):
        if self.image != None:
            return
        assert self.seq != None
        if self.seq == ( '' ):
            self.image = '.'
        else:
            self.image = '/'.join( self.seq )

    def ensureSeqIsAvailable( self ):
        if self.seq != None:
            return
        assert self.image != None
        aSeq = self.image.split( '/' )
        # take care of duplicated slashes
        if len( aSeq ) > 2:
            aSeq[ 1: ] = [ anItem for anItem in aSeq[ 1: ] if anItem != '' ]
        self.seq = aSeq
        self.isCanonicalized = False

    def toFullID( self ):
        self.canonicalize()
        if len( self.seq ) == 1:
            raise RuntimeError( "A relative path that has only one component cannot be converted into Full ID" )
        superSystemPathComps = self.seq[ : -1 ]
        return FullID( consts.SYSTEM,
            superSystemPathComps == [ '' ] and '/' or \
                '/'.join( self.seq[ : -1 ] ),
            self[ -1 ] )

    def isRoot( self ):
        self.canonicalize()
        return self.seq == [ '', '' ]

    def isAbsolute( self ):
        self.ensureSeqIsAvailable()
        return len( self.seq ) > 1 and self[0] == ''

    def append( self, aPathRepr ):
        self.ensureSeqIsAvailable()
        aPathCompSeq = None
        if type( aPathRepr ) in ( str, unicode ):
            if aPathRepr == '':
                aPathCompSeq = []
            else:
                aPathCompSeq = aPathRepr.split( '/' )
        elif type( aPathRepr ) in ( list, tuple ):
            aPathCompSeq = aPathRepr
        elif isinstance( aPathRepr, SystemPath ):
            aPathCompSeq = aPathRepr.toTuple()

        if len( aPathCompSeq ) == 0:
            return self
        if aPathCompSeq[ 0 ] == '':
            raise ValueError( 'Cannot join absolute system path' )

        if self.image != None:
            aPathReprStr = self.image
        else:
            aPathReprStr = ''

        if len( self.seq ) >= 1 and self.seq[ -1 ] == '':
            self.seq.pop()
            aPathReprStr = aPathReprStr.rstrip( '/' )
        for aPathComp in aPathCompSeq:
            if aPathComp != '':
                self.seq.append( aPathComp )
                aPathReprStr += '/' + aPathComp
        self.image = aPathReprStr
        self.isCanonicalized = False
        return self

    def canonicalize( self ):
        if self.isCanonicalized:
            return
        self.ensureSeqIsAvailable()
        aSeq = []
        for aPathComp in self.seq:
            if aPathComp == '.':
                continue
            elif aPathComp == '..':
                aSeq.pop()
            else:
                aSeq.append( aPathComp )
        self.image = None
        self.seq = aSeq
        self.isCanonicalized = True

    def toTuple( self ):
        self.ensureSeqIsAvailable()
        return tuple( self.seq )

    def toAbsolute( self, aBaseSystemPath ):
        if not aBaseSystemPath.isAbsolute():
            raise ValueError( "Base system path is expected to be absolute, "\
                              + "`%s' given." % aBaseSystemPath )
        if self.isAbsolute():
            return self

        return SystemPath( aBaseSystemPath ).append( self )

    def toRelative( self, aBaseSystemPath ):
        if not aBaseSystemPath.isAbsolute():
            raise ValueError( "Base system path is expected to be absolute, "\
                              + "`%s' given." % aBaseSystemPath )
        if not self.isAbsolute():
            return self

        self.canonicalize()
        aBaseSystemPath.canonicalize()
        aBaseIter = aBaseSystemPath.__iter__()
        aPathComp = None
        anOffset = 0

        try:
            aPathComps = []
            while anOffset < len( self.seq ):
                aPathComp = aBaseIter.next()
                if aPathComp != self.seq[ anOffset ]:
                    aPathComps.append( aPathComp == '' and '.' or '..' )
                    break
                anOffset += 1
            try:
                while True:
                    aPathComps.append( aBaseIter.next() == '' and '.' or '..' )
            except StopIteration:
                pass
            return SystemPath( aPathComps + self.seq[ anOffset: ] )
        except StopIteration:
            return SystemPath( self.seq[ anOffset: ] )

    def createFullID( self, kind, anID ):
        return FullID( kind, self, anID )

    def createVariableFullID( self, anID ):
        return self.createFullID( consts.VARIABLE, anID )

    def createProcessFullID( self, anID ):
        return self.createFullID( consts.PROCESS, anID )

    def createSystemFullID( self, anID ):
        return self.createFullID( consts.SYSTEM, anID )

    def getSuperSystemPath( self ):
        self.canonicalize()
        if len( self.seq ) == 0:
            raise RuntimeError(
                'Cannot retrieve a super-system path from the root path '
                + 'or a relative path with only one compoment.' )
        return SystemPath( self.seq[0: -1] )

    def isAncestorOf( self, aSystemPath ):
        self.canonicalize()
        aSystemPath.canonicalize()
        numOfPathComps = len( self.seq )
        if self.seq == [ '', '' ]:
            if aSystemPath.seq == [ '', '' ]:
                return False
            else:
                numOfPathComps -= 1
        if numOfPathComps >= len( aSystemPath.seq ):
            return False
        for anIndex in range( 0, numOfPathComps ):
            if self.seq[ anIndex ] != aSystemPath.seq[ anIndex ]:
                return False
        return True

    def findCommonPath( self, aSystemPath ):
        self.canonicalize()
        aSystemPath.canonicalize()
        isAbsolute = self.seq[ 0 ] == ''
        if isAbsolute ^ ( aSystemPath.seq[ 0 ] == '' ):
            raise ValueError, 'Both referred system paths should be either absolute or relative'
        anIndex = 0
        numPathComps = min( len( self.seq ), len( aSystemPath.seq ) )
        if numPathComps > 1:
            if self.seq[ 1 ] == '': 
                return SystemPath( self.seq )
            elif aSystemPath.seq[ 1 ] == '':
                return SystemPath( aSystemPath.seq )
        for anIndex in range( isAbsolute, numPathComps ):
            if self.seq[ anIndex ] != aSystemPath.seq[ anIndex ]:
                break
        return SystemPath( self.seq[ 0: anIndex + 1 ] )

    def isParentOf( self, aSystemPath ):
        self.canonicalize()
        aSystemPath.canonicalize()
        numOfPathComps = len( aSystemPath.seq )
        if self.seq == [ '', '' ]:
            if aSystemPath.seq == [ '', '' ]:
                return False
            elif numOfPathComps == 2:
                return True
        if len( self.seq ) != numOfPathComps - 1:
            return False
        for anIndex in range( 0, numOfPathComps - 1 ):
            if self.seq[ anIndex ] != aSystemPath.seq[ anIndex ]:
                return False
        return True

    def __str__( self ):
        self.ensureImageIsAvailable()
        return self.image

    def __len__( self ):
        self.ensureSeqIsAvailable()
        return len( self.seq )

    def __getitem__( self, anIndex ):
        self.ensureSeqIsAvailable()
        return self.seq[ anIndex ]

    def __setitem__( self, anIndex, aValue ):
        self.ensureSeqIsAvailable()
        self.image = None
        self.seq[ anIndex ] = aValue

    def __iter__( self ):
        self.ensureSeqIsAvailable()
        return self.seq.__iter__()

    def __hash__( self ):
        self.ensureImageIsAvailable()
        return hash( self.image )

    def __eq__( self, theOther ):
        if not isinstance( theOther, SystemPath ):
            return False
        return str( self ) == str( theOther )

ROOT_SYSTEM_FULLID = FullID( consts.SYSTEM, '', '/' )

if __name__ == '__main__':
    print SystemPath( '/' ) == SystemPath('/')
    print SystemPath( '///' ).toFullID()
    print SystemPath( '/CELL' ).toFullID()
    print FullID( 'System', '/', '' ) == FullID( 'System:/:' )
    print FullPN( 'System', '/', '', 'Size' ) in [ FullPN( FullID( 'System:/:' ), 'Size' ) ]
    print FullID( 'System', '/', '' )
    print FullID( 'System', '/', '' ).getSuperSystemPath()
    print hash( FullID( 'System', '/', '' ) )
    print SystemPath( '/' ).isParentOf( SystemPath( '/TEST' ) )
    print SystemPath( '/' ).isParentOf( SystemPath( '/' ) )
    print SystemPath( '/TEST' ).isParentOf( SystemPath( '/TEST' ) )
    print SystemPath( '/TEST' ).isParentOf( SystemPath( '/TEST/TEST' ) )
    print SystemPath( '/TEST' ).isParentOf( SystemPath( '/TEST/TEST/TEST' ) )
    print SystemPath( '/TEST' ).toRelative( SystemPath( '/ABC' ) )
    print SystemPath( '/TEST' ).toRelative( SystemPath( '/TEST' ) )
    print SystemPath( '/TEST' ).toRelative( SystemPath( '/TEST/TEST/ABC' ) )
    print SystemPath( '/test' ).toRelative( SystemPath( '/TEST/TEST/ABC' ) )
    print SystemPath( '/TEST/TEST/ABC' ).toRelative( SystemPath( '/TEST' ) )
    print SystemPath( '/TEST/TEST/ABC' ).toRelative( SystemPath( '/' ) )
    print SystemPath( '/' ).toRelative( SystemPath( '/TEST/TEST/ABC' ) )
    print SystemPath( '/' ).toRelative( SystemPath( '/' ) )
    print SystemPath( '/ABC/DEF' ).findCommonPath( SystemPath( '/ABC/DEF' ) )
    print SystemPath( '/ABC/DEF' ).findCommonPath( SystemPath( '/ABC' ) )
    print SystemPath( '/ABC/DEF' ).findCommonPath( SystemPath( '/' ) )
    print SystemPath( '/' ).findCommonPath( SystemPath( '/ABC/DEF' ) )
