
"""
A program for utilities for supporting analytical programs.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'util'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import string

from ecell.ecssupport import *

import numpy


RELATIVE_PERTURBATION = 0.001
ABSOLUTE_PERTURBATION = 1e-6


def allzero( a, err=0 ):
    '''
    return if the array has a nonzero element
    True means all elements in a that is zero
    a: (array) or (matrix)
    err: nonzero means abs of the value is larger than err
    return 0 or 1
    '''

    for element in a:
        if type( element ) == numpy.ArrayType or type( element ) == list:
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

    fullID = string.split( fullIDString, ':' )
    if fullID[ 0 ] == '':
        fullID[ 0 ] = 'Variable'
    try:
        fullID[ 0 ] = ENTITYTYPE_DICT[ fullID[ 0 ] ]
    except IndexError:
        raise ValueError( 'Invalid EntityType string (%s).' % fullID[ 0 ] )

    if fullID[ 0 ] != VARIABLE:
        raise ValueError( 'EntityType must be \'Variable\' (\'%s\' given)' % ( ENTITYTYPE_STRING_LIST[ fullID[ 0 ] ] ) )

    if fullID[ 1 ] == '.':
        processFullID = string.split( processID, ':' )
        if len( processFullID ) != 3:
            raise ValueError( 'processID has 3 fields. ( %d given )' % ( len( processFullID ) ) )
        fullID[ 1 ] = processFullID[ 1 ]

    validateFullID( fullID )
    return tuple( fullID )

# end of createVariableReferenceFullID


if __name__ == '__main__':

    pass
