#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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
A program for supporting the ECD file.
This program is the extension package for E-Cell System Version 3.
"""

__program__ = 'ecdsupport'
__version__ = '0.1'
__author__ = 'Kazunari Kaizu <kaizu@sfc.keio.ac.jp>'
__copyright__ = ''
__license__ = ''


import ecell.ECDDataFile

import numpy.fft
import numpy


class EcdSupport( ecell.ECDDataFile.ECDDataFile ):


    def __init__( self, aFileName=None ):
        '''
        supporting class for ecell.ECDDataFile.ECDDataFile
        aFileName: (str) ECD file name
        '''

        ecell.ECDDataFile.__init__( self, None, aFileName )

    # end of __init__


    def getFourierSpectrumData( self, anInterval, aStartTime=None ):
        '''
        get the Fourier spectrum 
        anInterval: (float) an interval
        aStartTime: (float) a start time
        return (list)
        '''

        aDiscretizedData = self.getFirstOrderDiscretizedData( anInterval, aStartTime )
        aValueList = []
        for aData in aDiscretizedData:
            aValueList.append( aData[ 1 ] )

        aSamplingTime = aDiscretizedData[ -1 ][ 0 ] - aDiscretizedData[ 0 ][ 0 ]
        
        aFourierClassList = numpy.fft.rfft( aValueList )
        anOrder = len( aFourierClassList )
        aSpectrumList = []
        aSpectrumList.append( [ 0.0, abs( aFourierClassList[ 0 ] ) / anOrder * 0.5 ] )
        for anIndex in range( 1, anOrder ):
            aSpectrumList.append( [ anIndex / aSamplingTime, abs( aFourierClassList[ anIndex ] ) / anOrder ] )

        return aSpectrumList

    # end of getFourierSpectrumData
    

    def getFirstOrderDiscretizedData( self, anInterval, aStartTime=None ):
        '''
        get the discretized data with constant interval
        first order interpolation
        anInterval: (float) an interval
        aStartTime: (float) a start time
        return (list)
        '''

        aFile = open( ecell.ECDDataFile.getFileName( self ), 'r' )
        aDataList = []

        while ( 1 ):

            line = aFile.readline()
            if not line:
                break

            if not ( len( line ) > 0 ):
                continue
            elif ( line[ 0 ] == '#' ):
                continue
            else:
                aData = line.split( '\t' )
                aPreviousTime = float( aData[ 0 ] )
                aPreviousValue = float( aData[ 1 ] )
                break

        if not aStartTime:
            aCurrentTime = aPreviousTime
        else:
            aCurrentTime = aStartTime
            
        while ( 1 ):

            line = aFile.readline()
            if not line:
                break
            
            aData = line.split( '\t' )
            aTime = float( aData[ 0 ] )
            aValue = float( aData[ 1 ] )

            if ( aTime > aPreviousTime ):
                k = ( aValue - aPreviousValue ) / ( aTime - aPreviousTime )
                while not ( aCurrentTime > aTime ):
                    anEstimatedValue = k * ( aCurrentTime - aTime ) + aPreviousValue
                    aDataList.append( [ aCurrentTime, anEstimatedValue ] )
                    aCurrentTime += anInterval

            aPreviousValue = aValue
            aPreviousTime = aTime

        aFile.close()

        return aDataList

    # end of getFirstOrderDiscretizedData


    def getFlexionPointList( self ):
        '''
        get a list of flexion points
        this function doesn\'t check the time and comments
        only for simple use
        return (list) of [ (float), (float), ( +1, -1, 0 ) ]
        '''

        aFlexionPointList = []
        aTrend = -1
        aPreviousTime = 0.0
        aPreviousValue = 0.0
        aPreviousTrend = 0

        aFile = open( ecell.ECDDataFile.getFileName( self ), 'r' )

        while ( 1 ):

            line = aFile.readline()
            if not line:
                break

            if not ( len( line ) > 0 ):
                continue
            elif ( line[ 0 ] == '#' ):
                continue
            else:
                aData = line.split( '\t' )

                aTime = float( aData[ 0 ] )
                aValue = float( aData[ 1 ] )

                line = aFile.readline()
                if not line:
                    break
                aData = line.split( '\t' )
                aPreviousTime = float( aData[ 0 ] )
                aPreviousValue = float( aData[ 1 ] )

                # reverse
                aPreviousTrend = cmp( aPreviousValue, aValue )
                
                break

        while ( 1 ):

            line = aFile.readline()
            if not line:
                break
            
            aData = line.split( '\t' )

            aTime = float( aData[ 0 ] )
            aValue = float( aData[ 1 ] )

            aTrend = cmp( aValue, aPreviousValue )

            if not ( aPreviousTrend == aTrend ):
                aFlexionPointList.append( [ aPreviousTime, aPreviousValue, aPreviousTrend ] )

            aPreviousTrend = aTrend
            aPreviousValue = aValue
            aPreviousTime = aTime

        if ( len( aFlexionPointList ) == 0 and aTrend == 0 ):
            aFlexionPointList.append( [ aPreviousTime, aPreviousValue, 0 ] )

        aFile.close()

        return aFlexionPointList

    # end of getFlexionPointList
    

# end of EcdSupport


if __name__ == '__main__':

    from ecdsupport import EcdSupport

    import sys


    def main( filename ):

        anEcdSupport = EcdSupport( filename )

        # print anEcdSupport.getFlexionPointList()
        # aDataList = anEcdSupport.getFirstOrderDiscretizedData( 0.1, 180 )
        aDataList = anEcdSupport.getFourierSpectrumData( 0.1, 180 )
        for aData in aDataList:
            print "%e\t%e" % ( aData[ 0 ], aData[ 1 ] )

    # end of main


    if ( len( sys.argv ) > 0 ):
        main( sys.argv[ 1 ] )
