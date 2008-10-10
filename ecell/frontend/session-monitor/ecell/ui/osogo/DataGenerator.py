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

from numpy import *
from random import *

import ecell.identifiers as identifiers

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

class DataGenerator:
    '''
    DataGenerator class
    get new data from the source, Logger or Core,
    and return data sets for plotting
    '''

    def __init__( self, aSession ):
        '''
        Constructor
        '''
        self.theSession = aSession
        self.theLoggerAdapter = LoggerManager( self.theSession )
        self.lastTime = 0.0
        self.dataProxy = {}

    def update( self ):
        self.theLoggerAdapter.update()

    def handleSessionEvent( self, event ):
        if event.type == 'simulation_updated':
            self.theLoggerAdapter.update()

    def requestNewData( self, aDataSeries, dataPointWidth ):
        '''
        update aDataSeries with new data points gotten from source
        '''
        dataList = zeros( (0,5), LOGGER_TYPECODE )
        xAxis = aDataSeries.getXAxis()
        aFullPN = aDataSeries.getFullPN()
        currentTime = self.theSession.getCurrentTime()

        if xAxis == "Time":
            dataList = aDataSeries.getAllData()
            if len( dataList ) > 0:
                lastTime = dataList[ len( dataList ) - 1 ][0]
            else:
                lastTime = currentTime
            dataRange =  currentTime - lastTime 

            requiredResolution = dataPointWidth
            if dataRange / requiredResolution > 100:
                requiredResolution = dataRange / 100
            dataList = self.theLoggerAdapter.getData(
                aFullPN, lastTime, currentTime, requiredResolution )
            size = len( dataList )
            if size == 0:
                dataList = zeros( (0,5), LOGGER_TYPECODE )
        else:
            assert isinstance( xAxis, identifiers.FullPN )
            # xaxis is fullpn, so this dataseries is used for phase plotting
            x = self.theLoggerAdapter.getCurrentValue( xAxis )
            y = self.theLoggerAdapter.getCurrentValue( aFullPN )
            dataList = zeros( ( 1, 5 ), LOGGER_TYPECODE )
            dataList[ 0 ][ 0 ] = x
            dataList[ 0 ][ 1 ] = y
            dataList[ 0 ][ 2 ] = y
            dataList[ 0 ][ 3 ] = y
            dataList[ 0 ][ 4 ] = y
            # do interpolation here
        self.lastTime = currentTime
        aDataSeries.addPoints( dataList )

    def __checkDataSeries( self, aDataSeries ):
        pass

    def requestData( self, aDataSeries, numberOfElements ):
        '''
        update aDataSeries with new data points gotten from whole Logger
        '''
        xAxis = aDataSeries.getXAxis()
        dataList = zeros( ( 0, 5 ), LOGGER_TYPECODE )
        aFullPN = aDataSeries.getFullPN()
        if xAxis == "Time":
            aLogger = self.theLoggerAdapter.getLogger( aFullPN )
            aStartTime = aLogger.getStartTime()
            anEndTime = aLogger.getEndTime()
            if anEndTime > aStartTime:
                requiredResolution = ( anEndTime - aStartTime ) / numberOfElements
            else:
                requiredResolution = 1
            dataList = aLogger.getData(
                aStartTime, anEndTime, requiredResolution )
        else:
            aRandom = Random()
            anXAxisLogger = self.theLoggerAdapter.getLogger( xAxis ) 
            aLogger = self.theLoggerAdapter.getLogger( aFullPN )
            yStartTime = aLogger.getStartTime()
            yWalker = LoggerWalker( aLogger )
            xWalker = LoggerWalker( anXAxisLogger )
            xstartpoint = xWalker.findPrevious( yStartTime )

            if  type( xstartpoint ) != int or xstartpoint != 1:
                writeCache = zeros( ( CACHE_INCREMENT, 5 ) )
                writeIndex = 0
                readIndex = 0
                xPoint = xWalker.getNext()
                while type( xPoint ) != int or xPoint != 1:
                    if aRandom.randint( 0, anXAxisLogger.getSize() ) < HISTORY_SAMPLE:
                        aTime = xPoint[DP_TIME]
                        yPoint1 = yWalker.findPrevious( aTime )
                        if type( yPoint1 ) == int and yPoint1 == 1:
                            break
                        newDataPoint = zeros( ( 5 ) )
                        newDataPoint[DP_TIME] = xPoint[DP_VALUE]
                        if yPoint1[DP_TIME] != aTime:
                            while True:
                                yPoint2 = yWalker.getNext( )
                                if type( yPoint2 ) == int and yPoint2 == 1:
                                    break
                                if yPoint2[DP_TIME] > yPoint1[DP_TIME]:
                                    break
                            if yPoint2 == 1:
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

            # do interpolation on X axis
        aDataSeries.replacePoints( dataList )

    def requestDataSlice( self, aDataSeries, startX, endX, requiredResolution ):
        '''
        request a slice from the Logger
        '''
        dataList = zeros( ( 0, 5 ), LOGGER_TYPECODE )
        xAxis = aDataSeries.getXAxis()
        aFullPN = aDataSeries.getFullPN()
        if xAxis == "Time":
            if not self.theLoggerAdapter.loggerExists( aFullPN ):
                dataList = zeros( ( 0, 5 ), LOGGER_TYPECODE )
            else:
                dataList = self.theLoggerAdapter.getData(
                    aFullPN, startX, endX, requiredResolution )
        aDataSeries.replacePoints( dataList )

class LoggerManager:
    def __init__ ( self, aSession ):
        self.theSession = aSession
        self.theManagedLoggers = {}

    def update( self ):
        for aCachedLogger in self.theManagedLoggers.values():
            aCachedLogger[ 1 ].update()

    def loggerExists( self, aFullPN ):
        return self.theManagedLoggers.has_key( str( aFullPN ) )

    def get( self, aFullPN ):
        aFullPNString = str( aFullPN )
        if aFullPNString not in self.theManagedLoggers.keys():
            self.theManagedLoggers[ aFullPNString ] = (
                self.theSession.createEntityStub( aFullPN.fullID ),
                CachedLogger( self.theSession.createLoggerStub( aFullPN ) )
                )
        return self.theManagedLoggers[ aFullPNString ]

    def getLogger( self, aFullPN ):
        return self.get( aFullPN )[ 1 ]

    def getData( self, aFullPN, start, end, interval ):
        return self.get( aFullPN )[ 1 ].getData( start, end, interval )

    def getStartTime( self, aFullPN ):
        return self.get( aFullPN )[ 1 ].getStartTime()

    def getEndTime( self, aFullPN ):
        return self.get( aFullPN )[ 1 ].getEndTime( )

    def getSize( self, aFullPN ):
        return self.get( aFullPN )[ 1 ].getSize()

    def getCurrentValue( self, aFullPN ):
        return self.get( aFullPN )[ 0 ].getProperty( aFullPN.propertyName )

class CachedLogger:
    def __init__( self, aStub ):
        self.theStub = aStub
        if not self.theStub.exists():
            self.theStub.create()
        self.theLoggerCacheList = []
        self.cachedTill = 0.0
        self.update()

    def update( self ):
        # calculate how many points should be read
        loggerSize = self.theStub.getSize()
        startTime = self.theStub.getStartTime()
        endTime = self.theStub.getEndTime()
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
            dataPoints = self.theStub.getData()
            valueColumn = take( dataPoints, (1, ), 1 )
            dataPoints = concatenate(
                ( dataPoints, valueColumn, valueColumn, valueColumn ), 1 )
            self.theLoggerCacheList[0].addPoints( dataPoints )
        else:
            while readStart < endTime:
                readEnd = min( readFrame, endTime - readStart ) + readStart
                dataPoints = self.theStub.getData( readStart, readEnd )
                valueColumn = take( dataPoints , (1, ), 1 )
                dataPoints = concatenate(
                    ( dataPoints, valueColumn, valueColumn, valueColumn ), 1 )
                self.theLoggerCacheList[ 0 ].addPoints( dataPoints )
                readStart = readEnd
        self.cachedTill = readStart            
        newPoints = self.theLoggerCacheList[ 0 ].getNewPoints()
        i = 1
        while len( newPoints ) > 1:
            if len(self.theLoggerCacheList) == i:
                self.theLoggerCacheList.append( LoggerCache() )
            self.theLoggerCacheList[i].addPoints( newPoints )
            newPoints = self.theLoggerCacheList[i].getNewPoints()
            i += 1
        
    def getData( self, start, end, interval = 1 ):
        if interval == 0:
            vectorLength = 1
        else:
            vectorLength = int( ( end - start ) / interval )
        i = len( self.theLoggerCacheList ) -1
        while i >= 0:
            aDistance = self.theLoggerCacheList[i].getAverageDistance()
            cacheStart = self.theLoggerCacheList[i].getStartTime()
            cacheEnd = self.theLoggerCacheList[i].getEndTime()
            if aDistance > 0 and \
               self.theLoggerCacheList[i].getSize() > vectorLength and \
               aDistance <= interval / 3:
                a=self.theLoggerCacheList[i].getData( start, end, interval )
                return a
            i -= 1
        # use logger
        a = self.theStub.getData( start, end, interval )
        return a

    def getStartTime( self ):
        return self.theStub.getStartTime()

    def getEndTime( self ):
        return self.theStub.getEndTime()

    def getSize( self ):
        return self.theStub.getSize()

class LoggerCache:
    def __init__( self ):
        self.theManagedLoggers  = zeros( ( 0, 5 ), LOGGER_TYPECODE )
        self.theResidue = zeros( ( 0, 5 ), LOGGER_TYPECODE ) # this contains the last unprocessed points
        self.cachedTill = 0
        self.residueEnd = 0
        self.cacheEnd = 0

    def getNewPoints( self ):
        #returns points that hasn't yet been cached
        cachedTill = self.cachedTill
        self.cachedTill = self.cacheEnd
        return  self.theManagedLoggers[ cachedTill:self.cacheEnd] 

    def addPoints( self, anArray ):
        #first add to residue
        requiredLength = len( anArray ) + self.residueEnd
        if len(self.theResidue) < requiredLength:
            self.theResidue = resize( self.theResidue, ( requiredLength, 5 )  )
        self.theResidue[self.residueEnd:self.residueEnd + len(anArray) ]= anArray[:]
        self.residueEnd += len( anArray )
        
        if len( self.theManagedLoggers ) == 0:
            self.theManagedLoggers = concatenate( ( self.theManagedLoggers, [ anArray[0] ] ) )
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
        if len( self.theManagedLoggers) < requiredLength:
            addition = (int(requiredLength / CACHE_INCREMENT) + 1 ) * CACHE_INCREMENT
            self.theManagedLoggers = resize( self.theManagedLoggers, ( len( self.theManagedLoggers) + addition, 5 ) )
        self.theManagedLoggers[self.cacheEnd:requiredLength] = anArray
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

    def getData( self, start, end, interval = 1 ):
        vectorLength = ( end - start ) / interval

        if vectorLength > int( vectorLength ):
            vectorLength += 1
        vectorLength = int(vectorLength )
        aVector = zeros( ( vectorLength, 5 ), LOGGER_TYPECODE )
        timeCounter = start
        lastPoint = self.theManagedLoggers[ self.cacheEnd - 1 ]
        for anIndex in range(0, vectorLength):
            startOffset = searchsorted( self.theManagedLoggers[ : self.cacheEnd, DP_TIME], timeCounter )
            if startOffset != 0:
                startOffset -= 1
            endOffset = searchsorted( self.theManagedLoggers[ startOffset:self.cacheEnd, DP_TIME], timeCounter + interval ) + startOffset
            if startOffset != self.cacheEnd:
                startOffset += 1
            aggregatedSlice =  self.theManagedLoggers[ startOffset: endOffset ] 

            if len(aggregatedSlice) > 0:
                lastPoint = self.__aggregateVector( aggregatedSlice )[0]
            aVector[anIndex] = lastPoint
            timeCounter += interval
        return aVector
            
    def getStartTime( self ):
        return self.theManagedLoggers[0, DP_TIME]

    def getEndTime( self ):
        return self.theManagedLoggers[ self.cacheEnd - 1, DP_TIME ]

    def getSize( self ):
        return  self.cacheEnd 

    def getAverageDistance( self ):
        if self.getSize() != 0:
            return ( self.getEndTime() - self.getStartTime() ) / self.getSize()
        else:
            return 0

class LoggerWalker:
    def __init__( self, aLogger ):
        self.theLogger = aLogger
        self.theStart = aLogger.getStartTime()
        self.theEnd = aLogger.getEndTime()
        size = aLogger.getSize()
        self.theAvgDistance = ( self.theEnd - self.theStart ) / size
        self.cachedTill = -1
        self.theCache = zeros( ( 0,5 ) )
        self.index = 0
        self.__readCache()

    def __readCache( self ):
        if self.cachedTill >= self.theEnd:
            return False

        if self.cachedTill == -1:
            readStart = self.theStart
        else:        
            readStart = self.cachedTill
        readEnd = min(
            self.theAvgDistance * int(READ_CACHE/2),
            self.theEnd - self.cachedTill ) + readStart
        newPoints = self.theLogger.getData( readStart, readEnd )
        if len( self.theCache) > 0:
            halfIndex = int(len(self.theCache) /2 )
            self.index = len( self.theCache ) - halfIndex
            self.theCache = concatenate(
                ( self.theCache[ halfIndex: ], newPoints ) )
        else:
            self.theCache = newPoints
            self.index = 0
        self.cachedTill = readEnd

        return True
        
    def getNext( self ):
        if self.index == len( self.theCache ):
            if not self.__readCache():
                return 1 # XXX: this kind of multiplexing is simply evil.
        aDataPoint = self.theCache[ self.index ]
        self.index += 1
        return aDataPoint
    
    def findPrevious( self, aTime ):
        #returns -1 aTime too small
        #returns 1 aTime too big
        nextIndex = searchsorted( self.theCache[:,DP_TIME], aTime )
        if nextIndex == len( self.theCache ):
            if not self.__readCache():
                return 1 # XXX: this kind of multiplexing is simply evil.
            return self.findPrevious( aTime )

        if aTime < self.theCache[ nextIndex ][DP_TIME]:
            if nextIndex > 0:
                nextIndex -= 1
            else:
                # smaller then what logger contains
                return -1 # XXX: this kind of multiplexing is simply evil.
        self.index = nextIndex
        return self.getNext()
