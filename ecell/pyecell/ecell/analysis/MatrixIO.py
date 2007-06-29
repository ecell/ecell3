#!/usr/bin/env python
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

"""
A program for utilities for supporting I/O.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'MatrixIO'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import re
import sys
from math import *

import numpy


def convertToDataString( value ):
    '''
    convert "Variable:/CELL/CYTOPLASM:S:Value"
    to "Variable__CELL_CYTOPLASM_S_Value"
    value: (str) ex. FullPN
    return (str)
    '''
    
    p = re.compile( '(/|:)' )
    if not type( value ) == str:
        raise TypeError( 'can not create DataString from %s type object' % type( value ) )
    else:
        return p.sub( '_', value )

# end of convertToDataString


def writeArrayWithMathematicaFormat( data, fout=sys.stdout ):
    '''
    write array or matrix to a file with mathematica format
    data: (array) or (matrix)
    fout: (str) output stream for the result
    '''

    size = len( data )
    fout.write( '{ ' )

    for c in range( size ):
        value = data[ c ]
        if type( value ) == numpy.ArrayType:
            writeArrayWithMathematicaFormat( value, fout )
        elif type( value ) == int or type( value ) == float:
            fout.write( '%.8e' % value )

        if c != size-1:
            fout.write( ', ' )

    fout.write( ' }' )

# end of writeArrayWithMathematicaFormat


def writeMatrix( outputFile, matrix, rowList=None, columnList=None ):
    '''
    write array or matrix to a file with CSV format
    it\'ll be slow than TableIO.writeArray because written in Python script
    outputFile: (str) or (file) output stream for the result
    matrix: (matrix)
    rowList: (list) row label list
    columnList: (list) column label list
    '''

    if type( outputFile ) == str:
        fout = open( outputFile, 'w' )
    elif type( outputFile ) == file:
        fout = outputFile
    else:
        return

    ( m, n ) = numpy.shape( matrix )

    if columnList:
        if rowList:
            for element in columnList:
                fout.write( ',%s' % element )
        elif len( columnList ) > 0:
            fout.write( '%s' % columnList[ 0 ] )
            for element in columnList[1:]:
                fout.write( ',%s' % element )
        fout.write( '\n' )

    for i in range( m ):
        if rowList:
            fout.write( '%s' % rowList[ i ] )
            for value in matrix[ i ]:
                if type( value ) == int:
                    fout.write( ',%d' % value )
                else:
                    fout.write( ',%.8e' % value )
        elif len( matrix[ i ] ) > 0:
            value = matrix[ i ][ 0 ]
            if type( value ) == int:
                fout.write( '%d' % value )
            else:
                fout.write( '%.8e' % value )
            for value in matrix[ i ][0:]:
                if type( value ) == int:
                    fout.write( ',%d' % value )
                else:
                    fout.write( ',%.8e' % value )

        fout.write( '\n' )

    if type( outputFile ) == str:
        fout.close()

# end of writeMatrix


def writeGraphViz( pathwayProxy, fout=sys.stdout ):
    '''
    write the stoichiometry matrix to a file with GraphViz Dot format.
    a Dot format file can be converted into some image file.
    % dot -Tps [model].dot -o [model].eps
    pathwayProxy: a PathwayProxy instance
    fout: (str) output stream for the result
    '''

    variableList = pathwayProxy.getVariableList()
    processList = pathwayProxy.getProcessList()

    stoichiometryMatrix = pathwayProxy.getStoichiometryMatrix()
    ( m, n ) = numpy.shape( stoichiometryMatrix )

    fout.write( 'digraph G {\n' )

    for processFullID in processList:
        processID = string.split( processFullID, ':' )[ 2 ]
        fout.write( '\t%s [shape=box,style=filled,color=lightgrey];\n' % processID )
    fout.write( '\n' )

    for j in range( n ):
        processID = string.split( processList[ j ], ':' )[ 2 ]

        substrateString = ''
        productString = ''

        for i in range( m ):
            variableID = string.split( variableList[ i ], ':' )[ 2 ]
            
            if stoichiometryMatrix[ i ][ j ] > 0:
                productString += '%s; ' % variableID
            elif stoichiometryMatrix[ i ][ j ] < 0:
                substrateString += '%s; ' % variableID

        attributeString = ''

        if substrateString != '':
            fout.write( '\t{ %s} -> %s%s;\n' % ( substrateString, processID, attributeString ) )
        if productString != '':
            fout.write( '\t%s -> { %s}%s;\n' % ( processID, productString, attributeString ) )

    fout.write( '}\n' )

# end of writeGraphViz


if __name__ == '__main__':

    pass
