#!/usr/bin/env python

from Numeric import *


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
        #self.hasLogger = 1
        # because having logger is varying from fpn to fpn
        self.lastTime = 0.0
        self.dataProxy = {}
        
    # end of __init__

    def hasLogger( self, aFullPNString ):
        return aFullPNString in self.__theSession.theSimulator.getLoggerList()

#    def setSource( self, aSource ):
 #      '''
#        aSource is either Logger or Core
#        Tracer and Plotter should take care that if source is Core,
#        then only requestNewData should be called
#        '''
#
#        if not ( aSource == 'Logger' ):
#            self.hasLogger = 1
#
#        elif ( aSource == 'Core' ):
#            self.hasLogger = 0
#
#        else:
#            # need some messages?
#            pass
#
    # end of setSource

# Source is varying from fpn to fpn depending whether it has logger, so source is not global


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
        pass
        
    def update( self ):
        pass
        
    def getData( fullPNString, start, end, interval ):
        pass
        
    def getStartTime( fullPNString ):
        pass
        
    def getEndTime( fullPNString ):
        pass
        
    def getSize( fullPNString ):
        pass

    

class CachedLogger:
    def __init__( self, aSession, fullPNString ):
        pass

    def update( self ):
        pass
        
    def getData( start, end, interval ):
        pass
        
    def getStartTime( ):
        pass
        
    def getEndTime(  ):
        pass

    def getSize( ):
        pass
        
