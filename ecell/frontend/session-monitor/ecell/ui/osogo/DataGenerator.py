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

from random import *
from numpy import *
from ecell.ecssupport import *
import gtk
import gtk.gdk

# aggregate so many points at any one level
AGGREGATE_POINTS = 200

# read no more than so many points from logger at once
READ_CACHE = 10000
HISTORY_SAMPLE = 2000

# cache increment
CACHE_INCREMENT = 10000

LOGGER_TYPECODE = 'd'

DP_TIME = 0
DP_VALUE = 1
DP_AVG = 2
DP_MAX = 4
DP_MIN = 3

class DataGenerator( object ):
    '''
    DataGenerator class
    get new data from the source, Logger or Core,
    and return data sets for plotting
    '''

    def __init__( self, aSession ):
        '''
        Constructor
        '''

        self.__theSession = aSession
        self.theLoggerAdapter = CachedLoggerAdapter( self.__theSession )
        self.lastTime = 0.0
        self.dataProxy = {}
        
    # end of __init__

    def update( self ):
        self.theLoggerAdapter.update()
        

    def hasLogger( self, aFullPNString ):
        return aFullPNString in self.__theSession.getLoggerList()



    def requestNewData( self, aDataSeries, dataPointWidth ):
        '''
        update aDataSeries with new data points gotten from source
        '''
        dataList = zeros( (0,5), LOGGER_TYPECODE )
        xAxis = aDataSeries.getXAxis()
        fullPNString = aDataSeries.getFullPNString()
        aFullID, aPropertyName = convertFullPNToFullID( createFullPN( fullPNString ) )
        currentTime = self.__theSession.getCurrentTime()

        if xAxis == "Time":

            if not self.hasLogger( fullPNString ):
                currentValue = self.__theSession.getEntityProperty( fullPNString )
                dataList = zeros( (1,5), LOGGER_TYPECODE )
                dataList[0][0] = currentTime
                dataList[0][1] = currentValue
                dataList[0][2] = currentValue
                dataList[0][3] = currentValue
                dataList[0][4] = currentValue
                
            else:
                dataList = aDataSeries.getAllData()
                if len( dataList ) > 0:
                    lastTime = dataList[ -1 ][ 0 ]
                else:
                    lastTime = 0.
                dataRange =  currentTime - lastTime 

                requiredResolution = dataPointWidth
                if (dataRange / requiredResolution) > 100:
                    requiredResolution = dataRange / 100
                dataList = self.theLoggerAdapter.getData( fullPNString, 
                            lastTime, currentTime, requiredResolution )
                
                # I havent yet updated the new logger code from CVS, but isn't changed to getDigest?
                
            size = len( dataList )
                
            if ( size == 0 ):
                dataList = zeros( (0,5), LOGGER_TYPECODE )
        

        else: #xaxis is fullpn, so this dataseries is used for phase plotting

            x = self.__theSession.getEntityProperty( xAxis )
            y = self.__theSession.getEntityProperty( fullPNString )
            dataList = zeros( (1,5), LOGGER_TYPECODE )
            dataList[0][0] = x
            dataList[0][1] = y
            dataList[0][2] = y
            dataList[0][3] = y
            dataList[0][4] = y
            # do interpolation here
        self.lastTime = currentTime
        
        aDataSeries.addPoints( dataList )

    # end of requestNewData

    def __checkDataSeries( self, aDataSeries ):
        pass


    def requestData( self, aDataSeries, numberOfElements ):
        '''
        update aDataSeries with new data points gotten from whole Logger
        '''

        xAxis = aDataSeries.getXAxis()
        dataList = zeros( (0,5), LOGGER_TYPECODE )
        fullPNString = aDataSeries.getFullPNString()
        if xAxis == "Time":
            if  self.hasLogger( fullPNString ) :
                aStartTime = self.theLoggerAdapter.getStartTime( fullPNString )
                anEndTime = self.theLoggerAdapter.getEndTime ( fullPNString )
                
                requiredResolution = ( anEndTime - aStartTime ) / numberOfElements
                dataList = self.theLoggerAdapter.getData( fullPNString, 
                                                          aStartTime,
                                                          anEndTime,
                                                          requiredResolution )

        else:
            aWindow = aDataSeries.thePlot.theWidget.get_ancestor( gtk.Window)
            if aWindow != None:
                aWindow.window.set_cursor( gtk.gdk.Cursor( gtk.gdk.WATCH ) )
                while gtk.events_pending():
                    gtk.main_iteration_do()
            aRandom = Random()

            if self.hasLogger( fullPNString ) and self.hasLogger( xAxis ):
                yStartTime = self.theLoggerAdapter.getStartTime( fullPNString )
                xAxisLogger = self.theLoggerAdapter.getLogger( xAxis )
                yWalker = LoggerWalker( self.theLoggerAdapter.getLogger( fullPNString ) )
                xWalker = LoggerWalker( xAxisLogger )
                xstartpoint = xWalker.findPrevious( yStartTime )

                if xstartpoint is not -1:
                    writeCache = zeros( ( CACHE_INCREMENT, 5 ) )
                    writeIndex = 0
                    readIndex = 0
                    xPoint = xWalker.getNext()
                    while xPoint is not 1:
                        if aRandom.randint( 0, xAxisLogger.getSize() ) < HISTORY_SAMPLE:
                            aTime = xPoint[DP_TIME]
                            yPoint1 = yWalker.findPrevious( aTime )
                            if yPoint1 is 1:
                                break
                            newDataPoint = zeros( ( 5 ) )
                            newDataPoint[DP_TIME] = xPoint[DP_VALUE]
                            if yPoint1[DP_TIME] != aTime:
                                while True:
                                    yPoint2 = yWalker.getNext()
                                    if yPoint2 is 1:
                                        break
                                    if yPoint2[DP_TIME] > yPoint1[DP_TIME]:
                                        break

                                if yPoint2 is 1:
                                    break
                                # interpolate
                                lowerTime = yPoint1[DP_TIME]
                                lowerValue = yPoint1[DP_VALUE]
                                upperTime = yPoint2[DP_TIME]
                                upperValue = yPoint2[DP_VALUE]
                                newDataPoint[DP_VALUE] = ( ( aTime - lowerTime ) * lowerValue + \
                                                        ( upperTime - aTime ) * upperValue ) / \
                                                           ( upperTime - lowerTime )

                            else:
                                newDataPoint[DP_VALUE] = yPoint1[DP_VALUE]
                            if writeIndex == len( writeCache):
                                writeCache = concatenate( ( writeCache, zeros( ( CACHE_INCREMENT, 5 ) ) ) )
                            writeCache[ writeIndex ] = newDataPoint
                            writeIndex += 1
                        xPoint = xWalker.getNext()
                    dataList = writeCache[:writeIndex] 
                    dataList [ :, 2 ] = dataList[ :, 1 ]
                    dataList [ :, 3 ] = dataList[ :, 1 ]
                    dataList [ :, 4 ] = dataList[ :, 1 ]

            if aWindow != None:
                aWindow.window.set_cursor( gtk.gdk.Cursor( gtk.gdk.TOP_LEFT_ARROW ) )

            # do interpolation on X axis
        aDataSeries.replacePoints( dataList )

    # end of requestData


    def requestDataSlice( self, aDataSeries, \
                          startX, endX, requiredResolution ):
        '''
        request a slice from the Logger
        '''
        dataList = zeros( (0,5), LOGGER_TYPECODE )
        xAxis = aDataSeries.getXAxis()
        fullPNString = aDataSeries.getFullPNString()
        if xAxis == "Time":

            if not ( self.hasLogger( fullPNString ) ):
                dataList =  zeros( (0,5 ), LOGGER_TYPECODE )
            else:
                dataList = self.theLoggerAdapter.getData( fullPNString, startX, endX, requiredResolution )

        else:
            pass
            #return
            # do Xaxis interpolation here for phase plotting
        aDataSeries.replacePoints( dataList )

    # end of requestDataSlice

# end of DataGenerator


class CachedLoggerAdapter:
    def __init__ ( self, aSession ):
        self.theSession = aSession
        self.theCachedLoggerDict = {}


    def update( self ):
        for aCachedLogger in self.theCachedLoggerDict.values():
            aCachedLogger.update()


    def getData( self, fullPNString, start, end, interval=None ):
        return self.getLogger( fullPNString ).getData( start, end, interval )


    def getStartTime( self, fullPNString ):
        return self.getLogger( fullPNString ).getStartTime()


    def getEndTime( self, fullPNString ):
        return self.getLogger( fullPNString ).getEndTime()


    def getSize( self, fullPNString ):
        return getLogger( fullPNString ).getSize()


    def getLogger( self, fullPNString ):
        aLogger = self.theCachedLoggerDict.get( fullPNString, None )
        if aLogger is None:
            aLogger = self.theCachedLoggerDict[ fullPNString ] = CachedLogger( self.theSession, fullPNString )
        return aLogger


class CachedLogger( object ):
    def __init__( self, aSession, fullPNString ):
        self.theLoggerStub = aSession.createLoggerStub( fullPNString )
        if fullPNString not in aSession.getLoggerList():
            self.theLoggerStub.setLoggerPolicy( aSession.getLogPolicyParameters() )
            self.theLoggerStub.create()
        self.theLoggerCacheList = []
        self.cachedTill = 0.0
        self.update()

    def update( self ):
        # calculate how many points should be read
        loggerSize = self.theLoggerStub.getSize()
        startTime = self.theLoggerStub.getStartTime()
        endTime = self.theLoggerStub.getEndTime()
        if loggerSize == 0:
            averageDistance = -1
        else:
            averageDistance = ( endTime - startTime ) / loggerSize
        # create first loggercache if doesnt exist
        if len( self.theLoggerCacheList ) == 0:
            self.theLoggerCacheList.append( LoggerCache() )
        if averageDistance <= 0 :
            #read all
            readFrame = endTime - startTime
        else:
            readFrame = averageDistance * READ_CACHE
        #call addpoints

        readStart = self.cachedTill
        if startTime == endTime:
            dataPoints = self.theLoggerStub.getData()

            valueColumn = take( dataPoints , (1, ), 1 )
            dataPoints = concatenate( ( dataPoints, valueColumn, valueColumn, valueColumn ) , 1 )

            self.theLoggerCacheList[0].addPoints( dataPoints )

        else:
            while readStart < endTime:
                readEnd = min( readFrame, endTime - readStart ) + readStart
                dataPoints = self.theLoggerStub.getData( readStart, readEnd )

                valueColumn = take( dataPoints , (1, ), 1 )
                dataPoints = concatenate( ( dataPoints, valueColumn, valueColumn, valueColumn ) , 1 )

                self.theLoggerCacheList[0].addPoints( dataPoints )
                readStart = readEnd

        self.cachedTill = readStart            
        newPoints = self.theLoggerCacheList[0].getNewPoints()
        i = 1
        while len( newPoints ) > 1:
            if len(self.theLoggerCacheList) == i:
                self.theLoggerCacheList.append(  LoggerCache() )
            self.theLoggerCacheList[i].addPoints( newPoints )
            newPoints = self.theLoggerCacheList[i].getNewPoints()
            i += 1
        
        
    def getData( self, start, end, interval=None ):
        if interval is not None:
            if start == end:
                return ndarray( shape=( 0, 5 ) )
            vectorLength = int( ( end - start ) / interval )
            i = len( self.theLoggerCacheList ) -1
            while i >= 0:
                aDistance = self.theLoggerCacheList[i].getAverageDistance()
                cacheStart = self.theLoggerCacheList[i].getStartTime()
                cacheEnd = self.theLoggerCacheList[i].getEndTime()
                if aDistance > 0 and \
                        self.theLoggerCacheList[i].getSize() > vectorLength and \
                        aDistance <= interval / 3:
#                    start >= cacheStart and end<= cacheEnd and 
#                return self.theLoggerCacheList[i].getData( start, end, interval )
                    return self.theLoggerCacheList[i].getData( start, end, interval )
                i -= 1
            return array( self.theLoggerStub.getData( start, end, interval ), copy=False)
        else:
            return array( self.theLoggerStub.getData( start, end ), copy=False )


    def getStartTime( self ):
        return self.theLoggerStub.getStartTime()


    def getEndTime( self ):
        return self.theLoggerStub.getEndTime()


    def getSize( self ):
        return self.theLoggerStub.getSize()


class LoggerCache( object ):
    
    def __init__( self ):
        self.theCache  = zeros( ( 0, 5 ), LOGGER_TYPECODE )
        self.theResidue = zeros( ( 0, 5 ), LOGGER_TYPECODE ) # this contains the last unprocessed points
        self.cachedTill = 0
        self.residueEnd = 0
        self.cacheEnd = 0


    def getNewPoints( self ):
        #returns points that hasn't yet been cached
        cachedTill = self.cachedTill
        self.cachedTill = self.cacheEnd
        return  self.theCache[ cachedTill:self.cacheEnd] 


    def addPoints( self, anArray ):
        #first add to residue

        requiredLength = len( anArray ) + self.residueEnd
        if len(self.theResidue) < requiredLength:
            self.theResidue = resize( self.theResidue, ( requiredLength, 5 )  )
        self.theResidue[self.residueEnd:self.residueEnd + len(anArray) ]= anArray[:]
        self.residueEnd += len( anArray )
        
        if len( self.theCache ) == 0:
            self.theCache = concatenate( ( self.theCache, [ anArray[0] ] ) )
        # process residue
        offset = 0
        while ( offset + AGGREGATE_POINTS ) <  self.residueEnd:
            newPoint = self.__aggregateVector( self.theResidue[ offset: offset + AGGREGATE_POINTS ] )
            # store processed in cache
            self.__addToCache ( newPoint ) 
            offset += AGGREGATE_POINTS
        # delete processed from residue
        residue = self.residueEnd - offset
        self.theResidue[0:residue] = self.theResidue[ offset:self.residueEnd]
        self.residueEnd = residue


    def __addToCache( self, anArray ):
        requiredLength = len( anArray ) + self.cacheEnd
        if len( self.theCache) < requiredLength:
            addition = (int(requiredLength / CACHE_INCREMENT) + 1 ) * CACHE_INCREMENT
            self.theCache = resize( self.theCache, ( len( self.theCache) + addition, 5 ) )
        self.theCache[self.cacheEnd:requiredLength] = anArray
        self.cacheEnd = requiredLength
        

    def __aggregateVector( self, anArray ):
        newPoint = zeros( ( 1, 5 ), LOGGER_TYPECODE )
        newPoint[0][ DP_TIME ] = anArray[ - 1 , DP_TIME ]
        newPoint[0][ DP_VALUE ] = anArray[ - 1 , DP_VALUE ]
        minPointIndex = argmin( anArray[ :, DP_MIN ] )
        newPoint[0][DP_MIN ] = anArray [ minPointIndex, DP_MIN ]
        maxPointIndex = argmax( anArray[ :, DP_MAX ] ) 
        newPoint[0][DP_MAX ] = anArray [ maxPointIndex, DP_MAX ]
        theSum = sum( anArray[ :, DP_AVG ]) 
        newPoint[0][ DP_AVG ] = theSum / len( anArray )
        return newPoint


    def getData( self, start, end, interval ):
        vectorLength = int( ceil( ( end - start ) / interval ) )
        aVector = zeros( ( vectorLength, 5 ), LOGGER_TYPECODE )
        timeCounter = start
        lastPoint = self.theCache[ self.cacheEnd - 1 ]
        for anIndex in range(0, vectorLength):
            startOffset = searchsorted( self.theCache[ : self.cacheEnd, DP_TIME], timeCounter )
            if startOffset != 0:
                startOffset -= 1
            endOffset = searchsorted( self.theCache[ startOffset:self.cacheEnd, DP_TIME], timeCounter + interval ) + startOffset
            if startOffset != self.cacheEnd:
                startOffset += 1
            aggregatedSlice =  self.theCache[ startOffset: endOffset ] 

            if len(aggregatedSlice) > 0:
                lastPoint = self.__aggregateVector( aggregatedSlice )[0]
            aVector[anIndex] = lastPoint
            timeCounter += interval

        return aVector
            


    def getStartTime( self ):
        return self.theCache[0, DP_TIME]


    def getEndTime( self ):
        return self.theCache[ self.cacheEnd - 1, DP_TIME ]


    def getSize( self ):
        return  self.cacheEnd 


    def getAverageDistance( self ):
        if self.getSize() != 0:
            return ( self.getEndTime() - self.getStartTime() ) / self.getSize()
        else:
            return 0


class LoggerWalker( object ):
    def __init__( self, aLogger ):
        self.theLogger = aLogger
        self.theStart = aLogger.getStartTime()
        self.theEnd = aLogger.getEndTime()
        size = aLogger.getSize()
        self.theAvgDistance = ( self.theEnd - self.theStart ) / size
        self.cachedTill = -1
        self.theCache = zeros( ( 0, 5 ) )
        self.index = 0
        self.__readCache()

        
    def __readCache( self ):
        if self.cachedTill >= self.theEnd:
            return False

        if self.cachedTill == -1:
            readStart = self.theStart
        else:        
            readStart = self.cachedTill
        readEnd = min( self.theAvgDistance * int(READ_CACHE/2), self.theEnd - self.cachedTill ) + readStart
        newPoints = self.theLogger.getData( readStart, readEnd )
        if len( self.theCache ) > 0:
            halfIndex = int(len(self.theCache) /2 )
            self.index = len( self.theCache ) - halfIndex
            self.theCache = concatenate( ( self.theCache[ halfIndex: ], newPoints ) )
        else:
            self.theCache = newPoints
            self.index = 0
        self.cachedTill = readEnd

        return True

        
    def getNext( self ):
        if self.index == len( self.theCache ):
            if not self.__readCache():
                return 1
        aDataPoint = self.theCache[ self.index ]
        self.index += 1
        return aDataPoint
            
            
    
    def findPrevious( self, aTime ):
        #returns -1 aTime too small
        #returns 1 aTime too big
        nextIndex = searchsorted( self.theCache[:,DP_TIME], aTime )
        if nextIndex == len( self.theCache ):
            if not self.__readCache():
                return 1
            return self.findPrevious( aTime )

        if aTime < self.theCache[ nextIndex ][DP_TIME]:
            if nextIndex > 0:
                nextIndex -= 1
            else:
                # smaller then what logger contains
                return -1
        self.index = nextIndex
        return self.getNext()
