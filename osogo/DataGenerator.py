#!/usr/bin/env python

from Numeric import *

# aggregate so many points at any one level
AGGREGATE_POINTS = 200

# read no more than so many points from logger at once
READ_CACHE = 10000


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

        self.__theSession = aSession
        self.lastTime = 0.0
        self.dataProxy = {}
        
    # end of __init__

    def hasLogger( self, aFullPNString ):
        return aFullPNString in self.__theSession.theSimulator.getLoggerList()



    def requestNewData( self, aDataSeries, requiredResolution ):
        '''
        update aDataSeries with new data points gotten from source
        '''
        dataList = zeros( (0,5) )
        xAxis = aDataSeries.getXAxis()
        fullPNString = aDataSeries.getFullPNString()
        currentTime = self.__theSession.theSimulator.getCurrentTime()

        if xAxis == "Time":

            if not self.hasLogger( fullPNString ):
                currentValue = self.__theSession.theSimulator.getEntityProperty( fullPNString )
                dataList = zeros( (1,5) )
                dataList[0][0] = currentTime
                dataList[0][1] = currentValue
                dataList[0][2] = currentValue
                dataList[0][3] = currentValue
                dataList[0][4] = currentValue
                
            else:
                dataList = aDataSeries.getAllData()
                lastTime = dataList[ len( dataList) -1 ][0]
                dataList = self.__theSession.theSimulator.getLoggerData( fullPNString, 
                            lastTime, currentTime,requiredResolution )
                
                # I havent yet updated the new logger code from CVS, but isn't changed to getDigest?
                
            size = len( dataList )
                
            if ( size == 0 ):
                dataList = zeros( (0,5) )
        

        else: #xaxis is fullpn, so this dataseries is used for phase plotting

            x = self.__theSession.theSimulator.getEntityProperty( xAxis )
            y = self.__theSession.theSimulator.getEntityProperty( fullPNString )
            dataList = zeros( (1,5) )
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
        dataList = zeros( (0,5) )
        fullPNString = aDataSeries.getFullPNString()
        if xAxis == "Time":

            if not ( self.hasLogger( fullPNString ) ):
    
                dataList =  zeros( (0,5 ) )
            else:

                aStartTime = self.__theSession.theSimulator.getLoggerStartTime( fullPNString )
                anEndTime = self.__theSession.theSimulator.getLoggerEndTime ( fullPNString )
                requiredResolution = ( anEndTime - aStartTime ) / numberOfElements
                dataList = self.__theSession.theSimulator.getLoggerData( fullPNString, 
                    aStartTime, anEndTime,   requiredResolution )
        else:
            pass
            #return
            # do interpolation on X axis

        aDataSeries.replacePoints( dataList )

    # end of requestData


    def requestDataSlice( self, aDataSeries, \
                          startX, endX, requiredResolution ):
        '''
        request a slice from the Logger
        '''
        dataList = zeros( (0,5) )
        xAxis = aDataSeries.getXAxis()
        fullPNString = aDataSeries.getFullPNString()
        if xAxis == "Time":

            if not ( self.hasLogger( fullPNString ) ):
    
                dataList =  zeros( (0,5 ) )
            else:
                dataList = self.__theSession.theSimulator.getLoggerData( fullPNString, startX, endX, requiredResolution )

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
        
    def getData( fullPNString, start, end, interval ):
        if fullPNString not in self.theCachedLoggerDict.keys():
            self.theCachedLoggerDict[ fullPNString ] = CachedLogger( self.theSession, fullPNString )
        self.theCachedLoggerDict[ fullPNString ].getData( start, end, interval )
            
            
        
    def getStartTime( fullPNString ):
        if fullPNString not in self.theCachedLoggerDict.keys():
            self.theCachedLoggerDict[ fullPNString ] = CachedLogger( self.theSession, fullPNString )
        self.theCachedLoggerDict[ fullPNString ].getStartTime( )
        
    def getEndTime( fullPNString ):
        if fullPNString not in self.theCachedLoggerDict.keys():
            self.theCachedLoggerDict[ fullPNString ] = CachedLogger( self.theSession, fullPNString )
        self.theCachedLoggerDict[ fullPNString ].getEndTime( )
        
    def getSize( fullPNString ):
        if fullPNString not in self.theCachedLoggerDict.keys():
            self.theCachedLoggerDict[ fullPNString ] = CachedLogger( self.theSession, fullPNString )
        self.theCachedLoggerDict[ fullPNString ].getSize( )

    

class CachedLogger:
    def __init__( self, aSession, fullPNString ):
        self.theFullPNString = fullPNString
        self.theSimulator = aSession.theSimulator
        if fullPNString not in self.theSimulator.getLoggerList():
            aSession.createLoggerWithPolicy( fullPNString )
        self.theLoggerCacheList = []
        self.cachedTill = 0.0
        self.update()

    def update( self ):
        # calculate how many points should be read
        
        # call 
        pass
        
    def getData( start, end, interval ):
        pass
        
    def getStartTime( self ):
        pass
        
    def getEndTime( self ):
        pass

    def getSize( self ):
        pass
        
class LoggerCache:
    
    def __init__( self ):
        self.theCache  = zeros( (0,5) )
        
        
    def addPoints( self, anArray ):
        pass
    
    def getData( self, start, end, interval ):
        pass
        
    def getStartTime( self ):
        pass
        
    def getEndTime( self ):
        pass

    def getSize( self ):
        pass
