#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
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

"""
A program for utilities for supporting analytical programs.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'util'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import re

from ecell.ecssupport import *

import numpy


RELATIVE_PERTURBATION = 1e-6
ABSOLUTE_PERTURBATION = 1e-6


def convertToDataString( aValue ):
    '''
    convert "Variable:/CELL/CYTOPLASM:S:Value"
    to "Variable__CELL_CYTOPLASM_S_Value"
    aValue: (str) ex. FullPN
    return (str)
    '''
    
    p = re.compile( '(/|:)' )
    if not type( aValue ) == str:
        raise TypeError( 'can not create DataString from %s type object' % type( aValue ) )
    else:
        return p.sub( '_', aValue )

# end of convertToDataString


def allzero( a, err=0 ):
    '''
    return if the array has a nonzero element
    True means all elements in a that is zero
    a: (array) or (matrix)
    err: nonzero means abs of the value is larger than err
    return 0 or 1
    '''

    for element in a:
        if type( element ) == numpy.ndarray or type( element ) == list:
            if not allzero( element ):
                return 0
        elif abs( element ) > err:
            return 0

    return 1

# end of allzero


def createIndependentGroupList( relationMatrix ):
    '''
    devide index into independent groups under the relation matrix
    relationMatrix: (matrix) connected or not
    return independentGroupList
    '''

    ( m, n ) = numpy.shape( relationMatrix )

    indexList = range( m )
    i = 0
    while i < len( indexList ):
        if allzero( relationMatrix[ indexList[ i ] ] ):
            indexList.pop( i )
        else:
            i += 1

    independentGroupList = []
    while len( indexList ) > 0:
        i = indexList.pop( 0 )
        relationList = relationMatrix[ i ].copy()
        independentGroupList.append( [ i ] )
        c = 0
        while c < len( indexList ):
            j = indexList[ c ]
            if allzero( numpy.logical_and( relationList, relationMatrix[ j ] ) ):
                relationList = numpy.logical_or( relationList, relationMatrix[ j ] )
                independentGroupList[ -1 ].append( indexList.pop( c ) )
            else:
                c += 1

    return independentGroupList

# end of createIndependentGroupList


def createVariableReferenceFullID( fullIDString, processID ):
    '''
    create a tuple from a variable\'s full ID
    support the abbreviation of the system path and the entity type
    fullIDString: (str) a variable full ID
    processID: (str) the ID of the process, having the variable reference
    return tuple( ENTITYTYPE, system path, variable name )
    '''

    fullID = fullIDString.split( ':' )
    if fullID[ 0 ] == '':
        fullID[ 0 ] = 'Variable'
    try:
        fullID[ 0 ] = ENTITYTYPE_DICT[ fullID[ 0 ] ]
    except IndexError:
        raise ValueError( 'Invalid EntityType string (%s).' % fullID[ 0 ] )

    if fullID[ 0 ] != VARIABLE:
        raise ValueError( 'EntityType must be \'Variable\' (\'%s\' given)' % ( ENTITYTYPE_STRING_LIST[ fullID[ 0 ] ] ) )

    if fullID[ 1 ] == '.':
        processFullID = processID.split( ':' )
        if len( processFullID ) != 3:
            raise ValueError( 'processID has 3 fields. ( %d given )' % ( len( processFullID ) ) )
        fullID[ 1 ] = processFullID[ 1 ]

    validateFullID( fullID )
    return tuple( fullID )

# end of createVariableReferenceFullID


if __name__ == '__main__':

    pass
