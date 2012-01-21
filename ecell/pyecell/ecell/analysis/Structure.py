#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
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

"""
A program for static structural analysis, such as stoichiometric analysis or flux balance analysis
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'Structure'
__version__ = '1.0'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import copy
import numpy


def generateFullRankMatrix( inputMatrix ):
    '''
    do Gaussian elimination and return the decomposed matrices
    inputMatrix: (matrix)
    return ( linkMatrix, kernelMatrix, independentList )
    '''

    reducedMatrix = inputMatrix.copy()
    ( m, n ) = numpy.shape( reducedMatrix )
    pivotMatrix = numpy.identity( m, float )

    dependentList = range( m )
    independentList = []
    skippedList = []
    skippedBuffer = []

    for j in range( n ):

        if len( dependentList ) == 0:
            break
        
        maxIndex = dependentList[ 0 ]
        maxElement = reducedMatrix[ maxIndex ][ j ]
        for i in dependentList:
            if abs( reducedMatrix[ i ][ j ] ) > abs( maxElement ):
                maxIndex = i
                maxElement = reducedMatrix[ i ][ j ]

        if abs( maxElement ) != 0.0:

            reducedMatrix[ maxIndex ] *= 1.0 / maxElement
            pivotMatrix[ maxIndex ] *= 1.0 / maxElement

            for i in range( m ):
                if i != maxIndex:
                    k = reducedMatrix[ i ][ j ]
                    reducedMatrix[ i ] -=  k * reducedMatrix[ maxIndex ]
                    pivotMatrix[ i ] -=  k * pivotMatrix[ maxIndex ]

            if len( skippedBuffer ) > 0:
                skippedList.extend( skippedBuffer )
                skippedBuffer = []

            dependentList.remove( maxIndex )
            independentList.append( maxIndex )
            
        else:
            skippedBuffer.append( j )

    linkMatrix = numpy.identity( m, float )
    for i in dependentList:
        linkMatrix[ i ] -= pivotMatrix[ i ]

    rank = len( independentList )
    kernelMatrix = numpy.zeros( ( n, n - rank ), float )
    parsedRank = rank + len( skippedList )
    reducedMatrix = numpy.take( reducedMatrix, range( parsedRank, n ) + skippedList, 1 )

    cnt1 = 0
    cnt2 = 0
    for i in range( parsedRank ):
        if len( skippedList ) > cnt1 and skippedList[ cnt1 ] == i:
            kernelMatrix[ i ][ n-parsedRank+cnt1 ] = 1.0
            cnt1 += 1
        else:
            numpy.put( kernelMatrix[ i ], range( n - rank ), -reducedMatrix[ independentList[ cnt2 ] ] )
            cnt2 += 1

    for i in range( n - parsedRank ):
        kernelMatrix[ i + parsedRank ][ i ] = 1.0

    independentList = numpy.sort( independentList )

    return ( numpy.take( linkMatrix, independentList, 1 ), kernelMatrix, independentList )

# end of generateFullRankMatrix


def printmat( stoichiometryList, modeList, reversibilityList ):
    '''
    '''
    
    for i in range( len( reversibilityList ) ):
        sys.stdout.write( '%d : ' % reversibilityList[ i ] )
        for j in range( len( stoichiometryList[ i ] ) ):
            sys.stdout.write( '%+2d ' % stoichiometryList[ i ][ j ] )
        sys.stdout.write( ' : ' )
        for j in range( len( modeList[ i ] ) ):
            sys.stdout.write( '%+2d ' % modeList[ i ][ j ] )
        sys.stdout.write( '\n' )
    sys.stdout.write( '\n' )

# end of printmat


def __checkModeDependency1( modeList, modeArray1, modeArray2=None ):
    '''
    check the mode dependency between modeArray1 and modeList
    S(m^{j}_i) \cap S(m^{j}_m) \not\subseteq S(m^{j+1}_l) 
    modeList: (list) m^{j+1}
    modeArray1: (array) m^{j}_i
    modeArray2: (array) m^{j}_m
    return 1 or 0
    '''

    if modeArray2 == None:
        modeArray2 = numpy.zeros( len( modeArray1 ) )

    if len( modeList ) == 0:
        return 0

    for targetModeArray in modeList:

        dependency = 1
        equality = 1
        for j in range( len( targetModeArray ) ):

            if  targetModeArray[ j ] != 0:
                if modeArray1[ j ] == 0 and modeArray2[ j ] == 0:
                    dependency = 0
                    equality = 0
                    break
            elif modeArray1[ j ] != 0 or modeArray2[ j ] != 0:
                equality = 0

        if dependency == 1 and equality == 0:
            return 1
            
    return 0

# end of checkModeDependency1


def __checkModeDependency2( modeList, modeArray ):
    '''
    check the mode dependency between modeArray and modeList
    S(m^{j+1}_l) \not\subset S(m^{j}_i) \cap S(m^{j}_m)
    modeList: (list) m^{j}
    modeArray: (array) m^{j+1}_l
    return 1 or 0
    '''

    if len( modeList ) == 0:
        return 0

    for i in range( len( modeList ) ):
        for m in range( i+1, len( modeList ) ):

            dependency = 1
            equality = 1
            for j in range( len( modeArray ) ):
                if  modeArray[ j ] == 0:
                    if modeList[ i ][ j ] != 0 or modeList[ m ][ j ] != 0:
                        dependency = 0
                        equality = 0
                        break

                elif modeList[ i ][ j ] == 0 and modeList[ m ][ j ] == 0:
                    equality = 0

            if dependency == 1 and equality == 0:
                return 1

    return 0

# end of checkModeDependency2


def generateElementaryFluxMode( stoichiometryMatrix, reversibilityList ):
    '''
    generate Elementary Flux Mode (EFM)
    stoichiometryMatrix: (matrix)
    reversibilityList: (list) indicate process reversibilities
    return modeList
    '''

    # rowSize is the number of processes,
    # and columnSize is the number of variables
    transStoichiometryMatrix = numpy.transpose( stoichiometryMatrix )
    ( rowSize, columnSize ) = numpy.shape( transStoichiometryMatrix )

    stoichiometryList = []
    modeList = []
    reversibilityList = copy.copy( reversibilityList )
    irreversibleOffset = 0

    for i in range( rowSize ):
        if reversibilityList[ i ] == 1:
            stoichiometryList.insert( 0, transStoichiometryMatrix[ i ].copy() )
            modeList.insert( 0, numpy.zeros( rowSize ) )
            modeList[ 0 ][ i ] = 1
            reversibilityList.insert( 0, reversibilityList.pop( i ) )
            irreversibleOffset += 1
        else:
            stoichiometryList.append( transStoichiometryMatrix[ i ].copy() )
            modeList.append( numpy.zeros( rowSize ) )
            modeList[ i ][ i ] = 1

    for j in range( columnSize ):

        # printmat( stoichiometryList, modeList, reversibilityList )

        newStoichiometryList = []
        newModeList = []
        newReversibilityList = []
        newIrreversibleOffset = 0

        for i in range( irreversibleOffset ):
            if stoichiometryList[ i ][ j ] == 0:
                newStoichiometryList.append( copy.copy( stoichiometryList[ i ] ) )
                newModeList.append( copy.copy( modeList[ i ] ) )
                newReversibilityList.append( 1 )
                newIrreversibleOffset += 1

        for i in range( irreversibleOffset ):
            if stoichiometryList[ i ][ j ] == 0:
                continue

            for m in range( i+1, irreversibleOffset ):
                if stoichiometryList[ m ][ j ] == 0:
                    continue
                
                if __checkModeDependency1( newModeList,\
                                           modeList[ i ], modeList[ m ] ) == 0:

                    coef_i = stoichiometryList[ i ][ j ]
                    coef_m = stoichiometryList[ m ][ j ]

                    newStoichiometryList.append( coef_i * stoichiometryList[ m ] - coef_m * stoichiometryList[ i ] )
                    newModeList.append( coef_i * modeList[ m ] - coef_m * modeList[ i ] )
                    newReversibilityList.append( 1 )
                    newIrreversibleOffset += 1
                    conditionCheck = 0

        for i in range( irreversibleOffset, len( modeList ) ):
            if stoichiometryList[ i ][ j ] == 0:
                newStoichiometryList.append( copy.copy( stoichiometryList[ i ] ) )
                newModeList.append( copy.copy( modeList[ i ] ) )
                newReversibilityList.append( 0 )

        for i in range( irreversibleOffset, len( modeList ) ):
            if stoichiometryList[ i ][ j ] == 0:
                continue

            for m in range( i+1, len( modeList ) ):
                if stoichiometryList[ m ][ j ] == 0:
                    continue
                
                if __checkModeDependency1( newModeList,\
                                           modeList[ i ], modeList[ m ] ) == 0:

                    coef_i = stoichiometryList[ i ][ j ]
                    coef_m = stoichiometryList[ m ][ j ]

                    if coef_i * coef_m < 0:
                        newStoichiometryList.append( numpy.fabs( coef_i ) * stoichiometryList[ m ] + numpy.fabs( coef_m ) * stoichiometryList[ i ] )
                        newModeList.append( numpy.fabs( coef_i ) * modeList[ m ] + numpy.fabs( coef_m ) * modeList[ i ] )
                        newReversibilityList.append( 0 )

        for i in range( irreversibleOffset, len( modeList ) ):
            if stoichiometryList[ i ][ j ] == 0:
                continue

            for m in range( irreversibleOffset ):
                if stoichiometryList[ m ][ j ] == 0:
                    continue
                
                if __checkModeDependency1( newModeList,\
                                           modeList[ i ], modeList[ m ] ) == 0:

                    coef_i = stoichiometryList[ i ][ j ]
                    coef_m = stoichiometryList[ m ][ j ]

                    # i must be irreversible. this should be modified.
                    if coef_i * coef_m > 0:
                        coef_i = -abs( coef_i )
                        coef_m = abs( coef_m )
                    else:
                        coef_i = abs( coef_i )
                        coef_m = abs( coef_m )

                    newStoichiometryList.append( coef_i * stoichiometryList[ m ] + coef_m * stoichiometryList[ i ] )
                    newModeList.append( coef_i * modeList[ m ] + coef_m * modeList[ i ] )
                    newReversibilityList.append( 0 )

        for l in range( len( newModeList )-1, -1, -1 ):
            if __checkModeDependency2( modeList, newModeList[ l ] ) == 1:

                newStoichiometryList.pop( l )
                newModeList.pop( l )
                newReversibilityList.pop( l )
                if newIrreversibleOffset > l:
                    newIrreversibleOffset -= 1

        stoichiometryList = copy.copy( newStoichiometryList )
        modeList = copy.copy( newModeList )
        reversibilityList = copy.copy( newReversibilityList )
        irreversibleOffset = newIrreversibleOffset

    # printmat( stoichiometryList, modeList, reversibilityList )

    return modeList

# end of generateElemetaryFluxMode


if __name__ == '__main__':

    from emlsupport import EmlSupport
    from Structure import *

    import sys
    import os


    def main( filename ):
        
        anEmlSupport = EmlSupport( filename )
        pathwayProxy = anEmlSupport.createPathwayProxy()
        
        stoichiometryMatrix = pathwayProxy.getStoichiometryMatrix()

        # print pathwayProxy.getProcessList()        
        # print pathwayProxy.getVariableList()

##         stoichiometryMatrix = array( [ [ 1., 0., 0., 0., 2. ], [ 0., 0., 1.0, 0., 1. ], [ 1.0, 0.0, 1.0, 0.0, 3.0 ], [ -2.0, 0.0, -2.0, 0.0, -6.0 ] ] )
##         stoichiometryMatrix = array( [ [ 1., 2., 3., 4. ], [ 1., 1., 1., 1. ], [ 2., 3., 4., 5. ] ] )
##         stoichiometryMatrix = array( [ [ 1., 2., 3. ], [ 0., 2., 10. ], [ 2., 3., 4. ] ] )

        print 'input matrix = '
        print stoichiometryMatrix

        ( linkMatrix, kernelMatrix, independentList ) = generateFullRankMatrix( stoichiometryMatrix )
        reducedMatrix = numpy.take( stoichiometryMatrix, independentList, 0 )

        print 'link matrix = '
        print linkMatrix
        print 'kernel matrix = '
        print kernelMatrix
        print 'reduced matrix = '
        print reducedMatrix

        # print 'reconstructed input matrix = '
        # print numpy.dot( linkMatrix, reducedMatrix )
        # print 'null space = '
        # print numpy.dot( stoichiometryMatrix, kernelMatrix )

        reversibilityList = pathwayProxy.getReversibilityList()
        
##         stoichiometryMatrix = numpy.transpose( array( [ [ 0, 0, 1, 0, 0 ], [ 0, -1, 0, 2, 0 ], [ -1, 0, 0, 0, 1 ], [ -2, 0, 2, 1, -1 ], [ 0, 0, 0, -1, 0 ], [ 1, 0, 0, 0, 0 ], [ 0, 1, -1, 0, 0 ], [ 0, -1, 1, 0, 0 ], [ 0, 0, 0, 0, -1 ] ], float ) )
##         reversibilityList = [ 1, 1, 1, 1, 0, 0, 0, 0, 0 ]

        print 'input list ='
        print reversibilityList

        print 'elementary flux mode list ='
        print generateElementaryFluxMode( stoichiometryMatrix, reversibilityList )
        
    # end of main
    

    if len( sys.argv ) > 1:
        main( sys.argv[ 1 ] )
    else:
        filename = '../../../../doc/samples/Heinrich/Heinrich.eml'
        main( os.path.abspath( filename ) )
