#!/usr/bin/env python

import gtk
import Numeric as nu
import gtk.gdk
import gobject
import re
from string import *
from math import *
import operator
from DataGenerator import *

from ecell.ecssupport import *

#class Plot,
 
#initializes graphics
#changes screen, handles color allocation when adding/removing traces
#clears plot areas on request, sets tityle


PLOT_VALUE_AXIS = "Value"
PLOT_TIME_AXIS = "Time"
TIME_AXIS = "Time"
PLOT_HORIZONTAL_AXIS = "Horizontal"
PLOT_VERTICAL_AXIS = "Vertical"
SCALE_LINEAR = "Linear"
SCALE_LOG10 = "Log10"
MODE_HISTORY = "history"
MODE_STRIP = "strip"

PEN_COLOR = "black"
BACKGROUND_COLOR = "grey"
PLOTAREA_COLOR = "light grey"

PLOT_MINIMUM_WIDTH = 400
PLOT_MINIMUM_HEIGHT = 250


ColorList=["pink","cyan","yellow","navy",
           "brown","white","purple","black",
           "green", "orange","blue","red"]
theMeasureDictionary = \
                     [['System', 'Size' , 'size'],
                      ['Variable', 'MolarConc', 'molar conc.' ],
                      ['Variable', 'NumberConc', 'number conc.' ],
                      ['Variable', 'TotalVelocity', 'total velocity' ],
                      ['Variable', 'Value', 'value'],
                      ['Variable', 'Velocity', 'velocity' ],
                      ['Process', 'Activity', 'activity'  ]]

            
def num_to_sci( num ):
    if (num<0.00001 and num>-0.00001 and num!=0) or num >999999 or num<-9999999:
        si=''
        if num<0: 
            si='-'

            num=-num
        ex=floor(log10(num))
        m=float(num)/pow(10,ex)
        t=join([si,str(m)[0:4],'e',str(ex)],'')
        return str(t[0:8])
                
    else:
        return str(num)[0:8]


def sign( num ):
    if int( num ) == 0:
        return 0
    return int( num / abs( num ) )

class Axis:
    def __init__( self, aParent, aType, anOrientation ):
        """ aParent = Plot instance
        aType = PLOT_TIME_AXIS or PLOT_VALUE_AXIS
        anOrientation = PLOT_HORIZONTAL_AXIS or PLOT_VERTICAL_AXIS
        aBindingBox = list of the [upperleftx, upperlefty, lowerleftx, lowerlefty]
        """
       
        self.theOrientation = anOrientation
        self.theScaleType = SCALE_LINEAR
        self.theMeasureLabel = ''
        self.theFrame = [0,0]
        self.theGrid = [0,0]
        self.theFullPNString = "Time"
        self.setType( aType )
        self.theParent = aParent


    def getFullPNString ( self ):
        return self.theFullPNString

    def setFullPNString ( self, aFullPNString ):
        self.theFullPNString = aFullPNString
        if aFullPNString == TIME_AXIS:
            self.setType ( PLOT_TIME_AXIS )
        else:
            self.setType( PLOT_VALUE_AXIS )


    def setType( self, aType ):
        self.theType = aType
        if self.theType == PLOT_VALUE_AXIS:
           self.rescaleTriggerMax = 0.90 
           self.rescaleTriggerMin = 0.0
           self.rescalingFrameMin = 0.1
           self.rescalingFrameMax = 0.9
        else:
            self.rescaleTriggerMax = 1
            self.rescaleTriggerMin = 0
            self.rescalingFrameMin = 0
            self.rescalingFrameMax = 0.5


    def getType ( self ):
        return self.theType


    def recalculateSize( self ):
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            self.theLength = self.theParent.thePlotArea[2]
            self.theBoundingBox =  [ 0, self.theParent.theOrigo[1] + 1, self.theParent.thePlotWidth, 
                                    10 + self.theParent.theAscent + self.theParent.theDescent ]
            self.theMaxTicks = int( self.theParent.thePlotWidth / 100 )
            self.theMeasureLabelPosition = [ self.theParent.thePlotAreaBox[2] , self.theBoundingBox[1] + self.theBoundingBox[3]-3 ]
        else:
            self.theLength = self.theParent.thePlotArea[3]
            self.theBoundingBox = [ 0, 0, self.theParent.theOrigo[0], self.theParent.theOrigo[1] + 5 ]
            self.theMaxTicks = int( self.theParent.thePlotHeight / 150 ) * 5
            self.theMeasureLabelPosition = [ self.theParent.theOrigo[0] - 50, 0 ]


    def clearLabelArea(self):
        self.theParent.drawBox( BACKGROUND_COLOR, self.theBoundingBox[0], self.theBoundingBox[1],
                                    self.theBoundingBox[2], self.theBoundingBox[3] )


    def draw(self):
        self.reprintLabels()
        self.drawMeasures()


    def printLabel( self, num ):
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            text = num_to_sci(num)
            x = self.convertNumToCoord( num )
            y = self.theParent.theOrigo[1] + 10
            self.theParent.drawText( PEN_COLOR, x - self.theParent.theFont.string_width( str(text) ) / 2, y, text )
            self.theParent.drawLine( PEN_COLOR, x, self.theParent.theOrigo[1], x, self.theParent.theOrigo[1] + 5 )
        else:
            text = num_to_sci(num)
            x = self.theParent.theOrigo[0] - 5 - self.theParent.theFont.string_width(text)
            y = self.convertNumToCoord( num )
            self.theParent.drawText( PEN_COLOR, x, y-7, text )
            self.theParent.drawLine( PEN_COLOR, self.theParent.theOrigo[0] - 5, y, self.theParent.theOrigo[0], y )


    def convertNumToCoord( self, aNumber ):
        if self.theScaleType == SCALE_LINEAR:
            relativeCoord = round( ( aNumber - self.theFrame[0] ) / float( self.thePixelSize ) )
        else:
            relativeCoord = round( (log10( aNumber ) - log10( self.theFrame[0] ) ) / self.thePixelSize )

        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            absoluteCoord = self.theParent.theOrigo[0] + relativeCoord
        else:
            absoluteCoord = self.theParent.theOrigo[1] - relativeCoord
        return absoluteCoord


    def convertCoordToNumber( self, aCoord ):
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            relativeCoord = aCoord - self.theParent.theOrigo[0]
        else:
            relativeCoord = self.theParent.theOrigo[1] - aCoord
        if self.theScaleType == SCALE_LINEAR:
            aNumber = float( relativeCoord ) * self.thePixelSize + self.theFrame[0]
        else:
            aNumber = pow(10, float ( relativeCoord ) * self.thePixelSize ) * self.theFrame[0]
        return aNumber


    def reprintLabels(self):
        #clears ylabel area
        self.clearLabelArea()
        tick=1
        if self.theScaleType == SCALE_LINEAR:
            for tick in range( int( self.theTickNumber + 1 ) ):
                tickvalue = self.theGrid[0] + tick * self.theTickStep
                self.printLabel( tickvalue )
        else:
            tickvalue = self.theGrid[1]
            while tickvalue > self.theGrid[0]:
                self.printLabel( tickvalue )
                tickvalue=tickvalue / self.theTickStep
            self.printLabel( self.theGrid[0] )


    def reframe( self ):  #no redraw!
        #get max and min from databuffers
        self.theParent.getRanges()
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            rangesMin = self.theParent.theRanges[ 2 ]
            rangesMax = self.theParent.theRanges[ 3 ]
        else:
            rangesMin = self.theParent.theRanges[ 0 ]
            rangesMax = self.theParent.theRanges[ 1 ]

        isStripTimeAxis = ( self.theParent.theStripMode == MODE_STRIP and self.theType == PLOT_TIME_AXIS )
        isOutOfBounds = self.isOutOfFrame( rangesMin, rangesMax )
        if isStripTimeAxis :

            self.theFrame[0] = rangesMin
            self.theFrame[1] = self.theFrame[0] + self.theParent.theStripInterval
            if ( self.theFrame[1] + self.theFrame[0] ) / 2  < rangesMax:
                self.theFrame[0] = rangesMax - self.theParent.theStripInterval / 2
                self.theFrame[1] = self.theFrame[0] + self.theParent.theStripInterval


        if self.theParent.theZoomLevel == 0 and not isStripTimeAxis:
            if rangesMin == rangesMax:
                if  rangesMin  == 0:
                    rangesMax = 1
                    rangesMin = -1
                else:
                    rangesMax = rangesMin + abs( rangesMin )
                    rangesMin = rangesMin - abs( rangesMin )

            #calculate yframemin, max
            if self.theScaleType == SCALE_LINEAR:
                aRange = ( rangesMax - rangesMin ) / \
                             ( self.rescalingFrameMax - self.rescalingFrameMin )
                self.theFrame[1] = rangesMax + ( 1 - self.rescalingFrameMax ) * aRange
                if rangesMin < 0:
                    self.theFrame[0] = rangesMin - ( self.rescalingFrameMin * aRange )
                else:
                    self.theFrame[0] = 0
                exponent = pow( 10, floor( log10( self.theFrame[1] - self.theFrame[0] ) ) )
                mantissa1 = ceil( self.theFrame[1] / exponent ) 
                mantissa0 = floor( self.theFrame[0] / exponent )
                sign0 = sign( mantissa0 )
                sign1 = sign( mantissa1 )
                mantissa0 = abs( mantissa0 )
                mantissa1 = abs( mantissa1 )
                if self.theFrame[1] <  0:
                    mantissa1 = 0.0
                else:
                    if mantissa1 <= 1.0:
                        mantissa1 = 1.0
                    elif mantissa1 <= 2.0:
                        mantissa1 = 2.0
                    elif mantissa1 <= 5.0:
                        mantissa1 = 5.0
                    else:
                        mantissa1 = 10.0

                self.theTickNumber = self.theMaxTicks
                halfTicks = int( self.theMaxTicks / 2 )
                if self.theFrame[0] < 0:
                    if mantissa0 <= 1.0:
                        mantissa0 = 1.0
                    elif mantissa0 <= 2.0:
                        mantissa0 = 2.0
                    elif mantissa0 <= 5.0:
                        mantissa0 = 5.0
                    else:
                        mantissa0 = 10.0

                if mantissa1 > mantissa0:
                    lesser=mantissa0
                    bigger=mantissa1
                else:
                    lesser=mantissa1
                    bigger=mantissa0

                tick_step = bigger / halfTicks
                lesser_step_no = ceil( lesser / tick_step )
                lesser = tick_step * float( lesser_step_no )
                lesser = int( lesser * 1000 ) / 1000.0
                self.theTickNumber = halfTicks + lesser_step_no

                if mantissa0 < mantissa1:
                    mantissa0 = lesser
                else:
                    mantissa1 = lesser    
                self.theFrame[1] = sign1 * mantissa1 * exponent
                self.theFrame[0] = sign0 * mantissa0 * exponent
                self.theTickStep = ( self.theFrame[1] - self.theFrame[0] ) / self.theTickNumber
            else: #log10 scaling

                if rangesMin <= 0 or rangesMax <= 0:
                    self.theParent.theOwner.theSession.message("non positive value in data, fallback to linear scale!\n")
                    self.theParent.changeScale( self.theOrientation, SCALE_LINEAR )
                    return

                self.theFrame[1] = pow( 10, ceil( log10( rangesMax ) ) )
                self.theFrame[0] = pow( 10, floor( log10( rangesMin ) ) )        
                diff = int( log10( self.theFrame[1] / self.theFrame[0] ) )
                if diff == 0:
                    diff=1
                if diff < self.theMaxTicks:
                    self.theTickNumber = diff
                    self.theTickStep = 10
                else:
                    self.theTickNumber = self.theMaxTicks
                    self.theTickStep = pow( 10, ceil( diff / self.theTickNumber ) )

            self.theGrid[0] = self.theFrame[0]
            self.theGrid[1] = self.theFrame[1]

        else: #if isOutOfBounds: # if zoomlevel > 0
            if self.theScaleType == SCALE_LINEAR:
                ticks=0
                if self.theFrame[1] == self.theFrame[0]:
                    self.theFrame[1] = self.theFrame[0] + abs( self.theFrame[0] ) *.1 + 1 
                exponent = pow( 10, floor( log10( self.theFrame[1] - self.theFrame[0] ) ) )

                while ticks < self.theMaxTicks / 2:
                    mantissa1 = floor( self.theFrame[1] / exponent )
                    mantissa0 = ceil( self.theFrame[0] / exponent )
                    ticks = mantissa1 - mantissa0
                    if ticks < self.theMaxTicks/2:
                        exponent=exponent/2

                    if ticks > self.theMaxTicks:
                        mantissa0 = ceil( mantissa0 / 2 ) * 2   
                        mantissa1 = floor( mantissa1 / 2 ) * 2
                        ticks = ( mantissa1 - mantissa0 ) / 2
                self.theTickNumber = ticks
                self.theGrid[1] = mantissa1 * exponent
                self.theGrid[0] = mantissa0 * exponent

                self.theTickStep = ( self.theGrid[1] - self.theGrid[0] ) / self.theTickNumber

                #scale is log
            else :   
                if self.theFrame[1] > 0 and self.theFrame[0] > 0:
                    self.theGrid[1] = pow( 10, floor( log10( self.theFrame[1] ) ) )
                    self.theGrid[0] = pow( 10, ceil( log10( self.theFrame[0] ) ) )      
                    diff = int( log10( self.theGrid[1] / self.theGrid[0] ) )
                    if diff == 0:
                        diff = 1
                    if diff < self.theMaxTicks:
                        self.theTickNumber = diff
                        self.theTickStep = 10
                    else:
                        self.theTickNumber = self.theMaxTicks
                        self.theTickStep = pow( 10, ceil( diff / self.theMaxTicks ) )
                else:
                    self.theParent.theOwner.theSession.message("non positive value in range, falling back to linear scale")
                    self.theParent.changeScale( self.theOrientation, SCALE_LINEAR )
                    return

        if self.theScaleType == SCALE_LINEAR:
            self.thePixelSize = float( self.theFrame[1] - self.theFrame[0] ) / self.theLength

        else:
            self.thePixelSize  =float( log10( self.theFrame[1] ) - log10( self.theFrame[0] ) ) / self.theLength
         #reprint_ylabels
         #self.reprintLabels()
        return 0


    def setScaleType( self, aScaleType ):
        self.theScaleType = aScaleType
        self.reframe()
        self.draw()


    def getScaleType( self ):
        return self.theScaleType


    def isOutOfFrame( self, aMin, aMax ):
        aRange = self.theFrame[1] - self.theFrame[0]
        return  ( aMin < self.theFrame[0] + aRange * self.rescaleTriggerMin )  or \
                  ( aMax > self.theFrame[0] + aRange * self.rescaleTriggerMax )



    def findMeasure( self, aFullPNString ):
        anArray = aFullPNString.split(':')
        aType = anArray[0]
        aProperty = anArray[3]
        aMeasure = "??"
        for aMeasureItem in theMeasureDictionary:
            if aMeasureItem[0]==aType and aMeasureItem[1]==aProperty:
                aMeasure= aMeasureItem[2]
                break
        return aMeasure


    def drawMeasures ( self ):
        # xmes is sec
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            if self.theType == PLOT_TIME_AXIS:
                self.theMeasureLabel = 'sec'
            elif self.theFullPNString == "":
                self.theMeasureLabel = "no trace"
            else:
                self.theMeasureLabel = self.findMeasure( self.theFullPNString )

        elif self.theParent.getSeriesCount() == 1:
            for aSeries in self.theParent.getDataSeriesList():
                if aSeries.isOn():
                    self.theMeasureLabel = self.findMeasure( aSeries.getFullPNString() )
                    break
        else:
            self.theMeasureLabel = "mixed traces"

        # add scale information
        scaleLabel = self.theScaleType + " scale"
        self.theMeasureLabel += "  " + scaleLabel

        # delete x area
        if self.theOrientation == PLOT_HORIZONTAL_AXIS:
            textLength = self.theParent.theFont.string_width( self.theMeasureLabel )
            self.theParent.drawBox(BACKGROUND_COLOR, self.theMeasureLabelPosition[0] - 200, 
                self.theMeasureLabelPosition[1], 200, 20)

            self.theParent.drawText(PEN_COLOR, self.theMeasureLabelPosition[0] - textLength, 
                self.theMeasureLabelPosition[1], self.theMeasureLabel )
        else:
            self.theParent.drawBox(BACKGROUND_COLOR, self.theMeasureLabelPosition[0], self.theMeasureLabelPosition[1], 140, 20)
    
            # drawText xmes
            self.theParent.drawText(PEN_COLOR, self.theMeasureLabelPosition[0], self.theMeasureLabelPosition[1],
                        self.theMeasureLabel )
                


class DataSeries:

    def __init__( self, aFullPNString, aDataSource, aPlot, aColor ):
        self.theFullPNString = aFullPNString
        self.isOnFlag = True
        self.theShortName = self.__getShortName ( self.theFullPNString )
        self.theLastXmax = None
        self.theLastXmin = None
        self.theLastYmin = None
        self.theLastYmax = None
        self.theLastY = None
        self.theLastX = None
        self.thePlot = aPlot
        self.theXAxis = "Time"
        self.theDataSource = aDataSource
        self.setColor( aColor )
        self.reset()

    def getXAxis ( self ):
        return self.thePlot.getXAxisFullPNString()

                  
    def getFullPNString( self ):
        return self.theFullPNString


    def reset( self ):
        self.theOldData = nu.zeros( ( 0 , 5 ) ) 
        self.theNewData = nu.zeros( ( 0 , 5 ) )

    def getPixBuf( self ):
        return self.thePixmap

    def setColor( self, aColor ):
        self.theColor = aColor
        self.thePixmap = self.thePlot.thePixmapDict[ aColor ]
            
    def getColor( self ):
        return self.theColor

    def getSource( self ):
        return self.theSource

    def setSource( self, aSource ):
        self.theSource = aSource

    def __getShortName(self, aFullPNString ):
        IdString = str( aFullPNString[ID] )
        PropertyString = str( aFullPNString[PROPERTY] )
        if PropertyString != 'Value':
           IdString += '/' + PropertyString[:2]
        return IdString
            
    def isOn( self ):
        return self.isOnFlag

    def switchOff (self ):
        self.isOnFlag = False
        
    def switchOn( self ):
        self.isOnFlag = True

    def addPoints( self, newPoints):
        """ newPoints: Numeric array of new points ( x, y, avg, max, min )"""

        self.theNewData = nu.concatenate( (self.theNewData, newPoints) )



    def replacePoints( self, newPoints):
        """ replace all datapoints """
        self.reset()
        self.addPoints( newPoints )

    def deletePoints( self, aThreshold ):
        """ delere datapoints where x<aThreshold suppose x is increasing monotonously
        use only for strip time plotting
        """
        idx = nu.searchsorted( nu.ravel( self.theOldData[:,0:1] ), aThreshold  ) 
        self.theOldData = nu.array( self.theOldData[ idx: ] ) 
                           
    def getOldData( self ):
        return self.theOldData

    def getAllData( self ):
        return nu.concatenate((self.theOldData, self.theNewData))

    def getNewData( self ):
        return self.theNewData

    

    def drawNewPoints( self ):
        self.drawTrace( self.getNewData() )
        self.theOldData = self.getAllData()
        self.theNewData = nu.zeros( ( 0,5 ) )

        

    def drawAllPoints( self ):
        #get databuffer, for each point draw
        self.theLastX = None
        self.theLastYmin = None
        self.theLastYmax = None
        self.theLastY = None
        self.drawTrace( self.getAllData() )
        self.theOldData = self.getAllData()
        self.theNewData = nu.zeros( ( 0,5 ) )


    def drawTrace( self, aPoints ):
        for aDataPoint in aPoints:
            #convert to plot coordinates

            if not self.isOn():
                return 0

            x=self.thePlot.theXAxis.convertNumToCoord( aDataPoint[ DP_TIME ] )
            y=self.thePlot.theYAxis.convertNumToCoord( aDataPoint[ DP_VALUE ] )
            ymax=self.thePlot.theYAxis.convertNumToCoord( aDataPoint[ DP_MAX ] )
            ymin=self.thePlot.theYAxis.convertNumToCoord( aDataPoint[ DP_MIN ] )

            #getlastpoint, calculate change to the last
            lastx = self.theLastX
            lasty = self.theLastY
            lastymax = self.theLastYmax
            lastymin = self.theLastYmin

            self.theLastX = x
            self.theLastY = y
            self.theLastYmax = ymax
            self.theLastYmin = ymin
            if x == lastx and y==lasty and ymax==lastymax and ymin==lastymin:
                continue

            if self.thePlot.getDisplayMinMax():
                self.drawMinMax( self.getColor(), x, ymax, ymin )

            self.drawPoint( self.getColor(), x, y, lastx, lasty )



    def drawMinMax(self, aColor, x, ymax, ymin):
        #first check x
        if x < self.thePlot.thePlotAreaBox[0] + 1 or x > self.thePlot.thePlotAreaBox[2] - 1:
            return
        #then check ymin<self.plotared[3], ymax>self.thePlotAreaBox[1]
        if ymax < self.thePlot.thePlotAreaBox[1] + 1 or ymin > self.thePlot.thePlotAreaBox[3] - 1:
            return

        # adjust frames
        if ymax >= self.thePlot.thePlotAreaBox[3]:
            ymax = self.thePlot.thePlotAreaBox[3] - 1
        if ymin <= self.thePlot.thePlotAreaBox[1] :
            ymin = self.thePlot.thePlotAreaBox[1] + 1

        # draw line
        self.thePlot.drawLine( aColor, x, ymin, x, ymax)



    def withinframes( self, point ):
        return point[0] < self.thePlot.thePlotAreaBox[2]  and point[0] > self.thePlot.thePlotAreaBox[0]  and\
               point[1] < self.thePlot.thePlotAreaBox[3]  and point[1] > self.thePlot.thePlotAreaBox[1] 


    def __adjustLimits( self, y0, y1, upLimit, downLimit ):
        if y0 < upLimit and y1 < upLimit:
            return None

        if y0 > downLimit  and y1 > downLimit:
            return None

        if y0 < upLimit:
            y0 = upLimit
        if y0 > downLimit:
            y0 = downLimit

        if y1 < upLimit:
            y1 = upLimit
        if y1 > downLimit:
            y1 = downLimit
        return [ y0, y1 ]


    def drawPoint(self, aColor, x, y, lastx, lasty ):
        #get datapoint x y values

        if lastx != None :
            dx = abs( lastx - x )
        else:
            dx = 0

        cur_point_within_frame=self.withinframes( [ x , y ] )
        last_point_within_frame=self.withinframes( [ lastx, lasty ] )

        if lasty != None:
            dy = abs( lasty - y )
        else:
            dy = 0
        if ( dx<2 and dy<2 ) or not self.thePlot.getConnectPoints():
            #draw just a point
            if cur_point_within_frame:
                self.thePlot.drawpoint_on_plot( self.getColor() ,x, y )

        elif self.thePlot.getConnectPoints():
            #draw line
            x0 = lastx
            y0 = lasty
            x1 = x
            y1 = y
            if cur_point_within_frame and last_point_within_frame:
                #if both points are in frame no interpolation needed
                pass
            else:
                upLimit = self.thePlot.thePlotAreaBox[1] + 1
                downLimit = self.thePlot.thePlotAreaBox[3] - 1
                leftLimit = self.thePlot.thePlotAreaBox[0] + 1
                rightLimit = self.thePlot.thePlotAreaBox[2] + 1
                #either current or last point out of frame, do interpolation
                    
                #interpolation section begins - only in case lastpoint or current point is off limits
                
                #there are 2 boundary cases x0=x1 and y0=y1
                if x0 == x1: 

                    if x0 < leftLimit or x0 > rightLimit:
                        return
                    #adjust y if necessary
                    result = self.__adjustLimits( y0, y1, upLimit, downLimit )
                    if result == None:
                        return
                    else:
                        y0, y1 = result

                elif y0 == y1: 

                    if y0 < downLimit or y0 > upLimit:
                        return
                    result = self.__adjustLimits( x0, x1, leftLimit, rightLimit )
                    if result == None:
                        return
                    else:
                        x0, x1 = result

                else:
                    #create coordinate equations
                    mx = float( y1 - y0 ) / float( x1 - x0 )
                    my = 1 / mx
                    xi = x0
                    yi = y0
                    xe = x1
                    ye = y1
                    #check whether either point is out of plot area
                        
                    #if x0 is out then interpolate x=leftside, create new x0, y0
                    if x0 < leftLimit:
                        if x1 < leftLimit:
                            return
                        x0 = leftLimit
                        y0 = yi + round( ( x0 - xi ) * mx )
                    #if y0 is still out, interpolate y=upper and lower side, 
                    #whichever x0 is smaller, create new x0
                    if y0 < upLimit or y0 > downLimit:
                        if y0 < upLimit:
                            #upper side
                            y0 = upLimit
                            x0 = xi + round( ( y0 - yi ) * my )
                        elif y0 > downLimit:
                            #lower side
                            y0 = downLimit
                            x0 = xi + round( ( y0 - yi ) * my )
                        if x0 < leftLimit or x0 > rightLimit or x0 < xi or x0 > x1:
                            return

                    #repeat it with x1 and y1, but compare to left side
                    if x1 > rightLimit:
                        if x0 > rightLimit:
                            return
                        x1 = rightLimit
                        y1 = yi + round( ( x1 - xi ) * mx )
                    #if y0 is still out, interpolate y=upper and lower side, 
                    #whichever x0 is smaller, create new x0
                    if y1 < upLimit or y1 > downLimit:
                        if y1 < upLimit:
                            #upper side
                            y1 = upLimit
                            x1 = xi + round( ( y1 - yi ) * my )
                        elif y1 > downLimit:
                            #lower side
                            y1 = downLimit
                            x1 = xi + round( ( y1 - yi ) * my )
                        if x1 < leftLimit or x1 > rightLimit or x1 < x0 or x1 > xe:
                            return

                #interpolation section ends
            self.thePlot.drawLine( self.getColor(), x0, y0, x1, y1 )


    def changeColor( self ):
        aColor = self.theColor
        self.theColor = ""
        self.thePlot.releaseColor( aColor )
        color_index = ColorList.index( aColor )
        color_index += 1
        if color_index == len( ColorList ):
            color_index = 0
        newColor = ColorList[ color_index ]
        self.thePlot.registerColor( newColor )
        self.setColor( newColor )
        self.thePlot.drawWholePlot()


class Plot:


    def __init__( self, owner, root, aDrawingArea ):
    # ------------------------------------------------------

        self.theGCColorMap={} 

        self.theSeriesMap = {} #list of displayed fullpnstrings
         
        self.theAvailableColors = ColorList[:]

        self.thePixmapDict={} #key is color, value pixmap
        self.theRoot=root[root.__class__.__name__]

        #creates widget
        self.theWidget = aDrawingArea

        if self.theWidget.get_ancestor( gtk.Window) != None:
            self.theWidget.unrealize()   
        #add buttonmasks to widget
        self.theWidget.set_events(gtk.gdk.EXPOSURE_MASK|gtk.gdk.BUTTON_PRESS_MASK|
                                    gtk.gdk.LEAVE_NOTIFY_MASK|gtk.gdk.POINTER_MOTION_MASK|
                                    gtk.gdk.BUTTON_RELEASE_MASK)
        if self.theWidget.get_ancestor( gtk.Window) != None:
            self.theWidget.realize()

        anAllocation = self.theWidget.get_allocation()
        self.thePlotWidth = max( anAllocation[2], PLOT_MINIMUM_WIDTH )
        self.thePlotHeight = max( anAllocation[3], PLOT_MINIMUM_HEIGHT )

        self.theColorMap=self.theWidget.get_colormap()
        self.theGCColorMap[ PEN_COLOR ] = self.getGC( PEN_COLOR )
        self.theGCColorMap[ BACKGROUND_COLOR ] = self.getGC( BACKGROUND_COLOR )
        self.theGCColorMap[ PLOTAREA_COLOR ] = self.getGC( PLOTAREA_COLOR )

        self.theStyle=self.theWidget.get_style()
        self.theFont=gtk.gdk.font_from_description(self.theStyle.font_desc)
        self.theAscent=self.theFont.ascent
        self.theDescent=self.theFont.descent
        self.thePixmapBuffer=gtk.gdk.Pixmap(self.theRoot.window,self.thePlotWidth,self.thePlotHeight,-1)
        self.theSecondaryBuffer=gtk.gdk.Pixmap(self.theRoot.window,self.thePlotWidth,self.thePlotHeight,-1)

        newpm = gtk.gdk.Pixmap(self.theRoot.window,10,10,-1)
        for aColor in ColorList:
            newgc = self.getGC ( aColor )
            self.theGCColorMap[ aColor ] = newgc
            newpm.draw_rectangle(newgc,gtk.TRUE,0,0,10,10)
            pb=gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB,gtk.TRUE,8,10,10)
            newpb=pb.get_from_drawable(newpm,self.theColorMap,0,0,0,0,10,10)
            self.thePixmapDict[ aColor ]=newpb

        self.theXAxis = Axis( self, PLOT_TIME_AXIS, PLOT_HORIZONTAL_AXIS )
        self.theYAxis = Axis ( self, PLOT_VALUE_AXIS, PLOT_VERTICAL_AXIS )
        self.recalculateSize()          
        self.theZoomLevel = 0
        self.theZoomBuffer = []
        self.theZoomKeyPressed = False
        self.theButtonTimeStampp = None
        self.theOwner = owner
        self.isControlShown = True
        #initializes variables
        self.theStripInterval = 1000
        self.doesConnectPoints = True
        self.doesDisplayMinMax = True
        # stripinterval/pixel
        self.doesRequireScaling=gtk.TRUE
        
        #initialize data buffers
        self.theStripMode = MODE_STRIP
        self.theZoomLevel = 0    

        self.totalRedraw()
        
        self.theWidget.connect('expose-event',self.expose)
        self.theWidget.connect('button-press-event',self.press)
        self.theWidget.connect('motion-notify-event',self.motion)
        self.theWidget.connect('button-release-event',self.release)

        
    def showControl( self, aState ):
        self.isControlShown = aState
        self.printTraceLabels()

        
    def getWidget( self ):
        return self.theWidget 

    def getMaxTraces( self ):
        return len( ColorList )

    def getXAxisFullPNString ( self ):
        return self.theXAxis.getFullPNString()

    def setXAxis( self, aFullPNString ):
        oldFullPN = self.theXAxis.getFullPNString()
        if oldFullPN != TIME_AXIS:
            #self.theSeriesMap[ oldFullPN ].switchOn()
            pass
        if aFullPNString == TIME_AXIS:
            self.doesConnectPoints = True
        else:
            #self.theSeriesMap[ aFullPNString ].switchOff()
            self.doesConnectPoints = False
        self.theXAxis.setFullPNString( aFullPNString )
       # take this out if phase plotting history is supported in datagenerator
        self.setStripMode( self.theStripMode )


    def changeScale(self, anOrientation, aScaleType ):
        #change variable
        if anOrientation == PLOT_HORIZONTAL_AXIS:
            anAxis = self.theXAxis
        else:
            anAxis = self.theYAxis
            
        anAxis.setScaleType( aScaleType )
        if anAxis.reframe() == None:
            return
        anAxis.draw()
        self.drawWholePlot()


    def totalRedraw( self ):
        self.clearPlot()
        self.drawAxes()
        self.drawWholePlot()


    def update( self ):
        if self.theZoomLevel > 0:
            return
        self.requestNewData()
        self.getRanges()

        redrawFlag = False
        if self.theYAxis.isOutOfFrame( self.theRanges[0], self.theRanges[1] ):
            self.theYAxis.reframe()
            self.theYAxis.draw()
            redrawFlag = True

        if self.theXAxis.isOutOfFrame( self.theRanges[2], self.theRanges[3] ):
            redrawFlag = True
            if self.theXAxis.getType() == PLOT_TIME_AXIS:
                if self.theStripMode == MODE_STRIP:
                    # delete half of the old data
                    self.shiftPlot()
                else:
                    self.requestData()
            self.theXAxis.reframe()

            self.theXAxis.draw()
        if redrawFlag:
            self.drawWholePlot()
        else:
            self.drawNewPoints()

    def drawNewPoints( self ):
        for aSeries in self.theSeriesMap.values():
            aSeries.drawNewPoints()

    def resetData( self ):
        for aSeries in self.getDataSeriesList():
            aSeries.replacePoints( zeros ( ( 0, 5) ) )

    def setStripMode(self, aMode):

        self.theStripMode = aMode
        if self.getXAxisFullPNString() == TIME_AXIS:
            self.requestData( )
            if aMode == MODE_STRIP:
        
                self.getRanges()
                self.theOwner.requestDataSlice( self.theRanges[3] - self.theStripInterval,
                    self.theRanges[3] - self.theStripInterval / 2, 
                    self.theStripInterval / self.theXAxis.theLength )
        else:
            if aMode == MODE_HISTORY:
                self.requestData()
            else:
                self.resetData()

        self.theZoomLevel = 0
        self.theZoomBuffer = []
        self.theZoomKeyPressed = False
        self.totalRedraw()


    def getDataSeriesList( self ):
        return self.theSeriesMap.values()
        
        
    def getDataSeries( self, aFullPNString ):
        return self.theSeriesMap[ aFullPNString ]


    def getDataSeriesNames( self ):
        return self.theSeriesMap.keys()
        
        
    def addTrace( self, aFPNStringList ):
        return_list = []
        if len( aFPNStringList ) == 0:
            return []
        for aFullPNString in aFPNStringList:
            #checks whether there's room for new traces
            #allocates a color

            aColor = self.allocateColor( )
            if aColor != None:
                aSeries = DataSeries( aFullPNString, self.theOwner, self, aColor )
                self.theSeriesMap[ aFullPNString ] = aSeries
                return_list.append( aFullPNString )

        #if mode is strip
        if self.theStripMode == MODE_STRIP:
            self.requestData( ) 

        elif self.theZoomLevel == 0:
            #if mode is history and zoom level 0, set xframes
            self.requestData( )

        else: #just get data from loggers
            self.requestDataSlice( self.theXAxis.theFrame[0], self.theXAxis.theFrame[1] )
        self.totalRedraw( )
        return return_list


    
    def removeTrace(self, FullPNStringList):
        #call superclass
        #redraw
        for fpn in FullPNStringList:
            self.releaseColor ( self.theSeriesMap[ fpn ].getColor() )
            self.theSeriesMap.__delitem__( fpn )
        self.totalRedraw()

  
    def getStripInterval(self):
        return self.theStripInterval
    

    def setStripInterval( self, newinterval ):
        #calulates new xframes, if there are more data in buffer
        self.theStripInterval = newinterval
        if self.theStripMode == MODE_STRIP:
            self.totalRedraw()


    def requestData(self ):
        self.theOwner.requestData( self.theXAxis.theLength * 2 )
        

    def requestDataSlice( self, aStart, anEnd ):
        self.theOwner.requestDataSlice( aStart, anEnd, ( anEnd - aStart ) / ( self.theXAxis.theLength*2) )

    def requestNewData ( self ):
        self.theOwner.requestNewData( self.getRequiredTimeResolution() )


    def getRequiredTimeResolution( self ):
        return ( self.theXAxis.theFrame[1] - self.theXAxis.theFrame[0] ) / (self.theXAxis.theLength * 2)
        
        
    def doConnectPoints( self, aBool ):
        self.doesConnectPoints = aBool

    def getConnectPoints( self ):
        return self.doesConnectPoints

    def doDisplayMinMax ( self, aBool ):
        self.doesDisplayMinMax = aBool

    def getDisplayMinMax( self ):
        return self.doesDisplayMinMax


    def isTimePlot( self ):
        return self.theXAxis.getType() == PLOT_TIME_AXIS


    def getGC ( self, aColor ):
        newgc = self.theRoot.window.new_gc()
        newgc.set_foreground( self.theColorMap.alloc_color( aColor ) )
        return newgc


    def recalculateSize(self):
        self.theOrigo=[70,self.thePlotHeight-35]
        self.thePlotArea=[self.theOrigo[0],30,\
            self.thePlotWidth-20-self.theOrigo[0],\
            self.theOrigo[1]-30]
        self.thePlotAreaBox=[self.thePlotArea[0],self.thePlotArea[1],\
            self.thePlotArea[2]+self.thePlotArea[0],\
            self.thePlotArea[3]+self.thePlotArea[1]]
        fontHeight = self.theAscent + self.theDescent
        self.persistentCoordArea = [200, 5, self.thePlotWidth / 2 - 100, fontHeight ]
        self.temporaryCoordArea = [ 100 + self.thePlotWidth / 2 , 5, self.thePlotWidth / 2 -100, fontHeight ]
        self.historyArea = [ 5, self.thePlotHeight - fontHeight, self.theFont.string_width( MODE_HISTORY ), fontHeight ]
        self.theXAxis.recalculateSize()
        self.theYAxis.recalculateSize()


    def expose(self, obj, event):

        obj.window.draw_drawable( self.thePixmapBuffer.new_gc(), self.thePixmapBuffer, event.area[0], event.area[1],
                                    event.area[0], event.area[1], event.area[2], event.area[3] )
        # check for resize
        anAllocation = self.theWidget.get_allocation()
        self.resize( anAllocation[2], anAllocation[3] )


    def allocateColor( self ):             
        #checks whether there's room for new traces
        if len( self.theAvailableColors ) > 0:
            #allocates a color
            allocated_color = self.theAvailableColors.pop()
            return allocated_color
        else:
            return None


    def registerColor( self, aColor ):
        if aColor in self.theAvailableColors:
            self.theAvailableColors.remove( aColor )


    def releaseColor(self, aColor ):
        #remove from colorlist
        for aSeries in self.theSeriesMap.values():
            if aSeries.getColor() == aColor:
                return
        self.theAvailableColors.insert( 0, aColor )


    def clearPlotarea(self):
        self.drawBox( PLOTAREA_COLOR, self.thePlotArea[0] + 1, self.thePlotArea[1] + 1,
                      self.thePlotArea[2] - 1, self.thePlotArea[3] -1 )
        

    def clearPlot(self):
        self.drawBox( BACKGROUND_COLOR, 0, 0, self.thePlotWidth, self.thePlotHeight )
        
        

    def drawAxes(self):
        # reframe too!!!
        self.theXAxis.reframe()
        self.theYAxis.reframe()
        self.theXAxis.draw()
        self.theYAxis.draw()

        
    def drawpoint_on_plot( self, aColor, x, y ):
        #uses raw plot coordinates!
        self.thePixmapBuffer.draw_point( self.theGCColorMap[aColor], int(x), int(y) )
        self.theWidget.queue_draw_area( int(x), int(y), 1, 1 )
    
        
    def drawLine(self, aColor, x0, y0, x1, y1):
        #uses raw plot coordinates!     
        self.thePixmapBuffer.draw_line(self.theGCColorMap[aColor],int(x0),int(y0),\
            int(x1),int(y1))
        self.theWidget.queue_draw_area(int(min(x0,x1)),int(min(y0,y1)),\
            int(abs(x1-x0)+1),int(abs(y1-y0)+1))
        return [x1,y1]

    def drawBox(self, aColor, x0,y0,width,height):
        #uses raw plot coordinates!
        self.thePixmapBuffer.draw_rectangle(self.theGCColorMap[aColor],gtk.TRUE,x0,y0,width,height)
        self.theWidget.queue_draw_area(x0,y0,width,height)

    def drawxorbox(self,x0,y0,x1,y1):
        newgc=self.thePixmapBuffer.new_gc()
        newgc.set_function(gtk.gdk.INVERT)
        self.thePixmapBuffer.draw_rectangle(newgc,gtk.TRUE,int(x0),int(y0),int(x1-x0),int(y1-y0))
        self.theWidget.queue_draw_area(int(x0),int(y0),int(x1),int(y1))

    def drawText(self,aColor,x0,y0,text):
            t=str(text)
            self.thePixmapBuffer.draw_text(self.theFont,self.theGCColorMap[aColor],int(x0),int(y0+self.theAscent),t)
            self.theWidget.queue_draw_area(int(x0),int(y0),self.theFont.string_width(t),self.theAscent+self.theDescent)


    def shiftPlot( self ):
        halfPoint =  self.theRanges[3] - int( self.theStripInterval / 2 ) 
        self.cutSeries( halfPoint )

    def cutSeries( self, aThreshold ):
        for aSeries in self.theSeriesMap.values():
            aSeries.deletePoints( aThreshold )


    def getRanges(self):
        self.theRanges = [ 0, 0, 0, 0 ]
        #search

        anArray = nu.reshape( nu.array([]),(0,5))
        for aDataSeries in self.theSeriesMap.values():
            if aDataSeries.isOn():
                anArray = nu.concatenate( ( anArray, aDataSeries.getAllData() ) )
        #init values
        if len( anArray ) > 0:
            self.theRanges[0] = anArray[ nu.argmin( anArray[:,DP_MIN] ), DP_MIN ] # minimum value 
            self.theRanges[1] = anArray[ nu.argmax( anArray[:,DP_MAX]), DP_MAX] #maximum value of all
            self.theRanges[2] = anArray[ nu.argmin( anArray[:,DP_TIME] ), DP_TIME ] # minimum time
            self.theRanges[3] = anArray[ nu.argmax( anArray[:,DP_TIME] ), DP_TIME ] # maximum time

  
    def press(self,obj, event):
        x = event.x
        y = event.y
        button=event.button
        #if button is 1
        if button==1:
            self.showPersistentCoordinates( x, y )
            # if inside plotarea, display 
            tstamp=event.get_time()
            if self.theButtonTimeStampp == tstamp: 
                if not self.isControlShown:
                    self.maximize()
                else:
                    self.minimize()
                return

            self.theButtonTimeStampp=tstamp                 
            if self.theStripMode==MODE_HISTORY and self.theXAxis.getFullPNString() == TIME_AXIS:
                #check that mode is history 
                self.theZoomKeyPressed=True
                self.x0 = x
                self.y0 = y
                self.x0 = max(self.thePlotArea[0],self.x0)
                self.y0 = max(self.thePlotArea[1],self.y0)
                self.x0 = min(self.thePlotArea[2]+self.thePlotArea[0],self.x0)
                self.y0 = min(self.thePlotArea[3]+self.thePlotArea[1],self.y0)
                self.x1 = self.x0
                self.y1 = self.y0
                self.realx0 = self.x0
                self.realx1 = self.x0
                self.realy0 = self.y0
                self.realy1 = self.y0
            #create self.x0, y0
        #if button is 3 and zoomlevel>0
        elif button==3:
            self.showMenu()
            #call zoomOut

    def showMenu( self ):
        theMenu = gtk.Menu()
        if self.theZoomLevel > 0:
            zoomUt = gtk.MenuItem( "Zoom out" )
            zoomUt.connect ("activate", self.__zoomOut )
            theMenu.append( zoomUt )
            theMenu.append( gtk.SeparatorMenuItem() )

        if self.isControlShown:
            guiMenuItem = gtk.MenuItem( "Hide Control" )
            guiMenuItem.connect ( "activate", self.__minimize_action )
        else:
            guiMenuItem = gtk.MenuItem( "Show Control" )
            guiMenuItem.connect ( "activate", self.__maximize_action )
         
        xToggle = gtk.MenuItem ( "Toggle X axis" )
        xToggle.connect( "activate", self.__toggleXAxis )
        yToggle = gtk.MenuItem ( "Toggle Y axis" )
        yToggle.connect( "activate", self.__toggleYAxis )
        #take this condition out if phase plotting works for history
        if self.theOwner.allHasLogger():
            if self.theStripMode == MODE_STRIP:
                toggleStrip = gtk.MenuItem("History mode")
            else:
                toggleStrip = gtk.MenuItem( "Strip mode" )
            toggleStrip.connect( "activate", self.__toggleStripMode )
            theMenu.append( toggleStrip )
            theMenu.append( gtk.SeparatorMenuItem() )   
        theMenu.append( xToggle )
        theMenu.append( yToggle )
        theMenu.append( gtk.SeparatorMenuItem() )
        theMenu.append( guiMenuItem )
        theMenu.show_all()
        theMenu.popup( None, None, None, 1, 0 )

    def __zoomOut( self, *args ):
        self.zoomOut()


    def __toggleStripMode( self, *args ):
        if self.theStripMode == MODE_STRIP:
            self.setStripMode( MODE_HISTORY )
        else:
            self.setStripMode( MODE_STRIP )


    def __toggleXAxis( self, *args ):
        if self.theXAxis.theScaleType == SCALE_LINEAR:
            self.changeScale( PLOT_HORIZONTAL_AXIS, SCALE_LOG10) 
        else:
            self.changeScale( PLOT_HORIZONTAL_AXIS, SCALE_LINEAR )

            
    def __toggleYAxis( self, *args ):
        if self.theYAxis.theScaleType == SCALE_LINEAR:
            self.changeScale( PLOT_VERTICAL_AXIS, SCALE_LOG10) 
        else:
            self.changeScale( PLOT_VERTICAL_AXIS, SCALE_LINEAR )
            

    def __minimize_action( self, *args ):
        self.minimize()

    def __maximize_action( self, *args ):
        self.maximize()

    def showPersistentCoordinates( self, x, y ):
        self.displayCoordinates( x, y, self.persistentCoordArea )

    def showTempCoordinates( self, x, y ):
        self.displayCoordinates( x, y, self.temporaryCoordArea )


    def displayCoordinates( self, x, y, aBox ):
        # displays coordinates at the top of chart
        aCoords = self.convertPlotCoordinates( x, y )
        # delete coord area
        self.drawBox( BACKGROUND_COLOR, aBox[0], aBox[1], aBox[2], aBox[3] )
        # write new coordinates
        if aCoords != None:
            text = num_to_sci( aCoords[0] ) + "  x  " + num_to_sci( aCoords[1] )
            self.drawText( PEN_COLOR, aBox[0], aBox[1], text )
            
  

    def convertPlotCoordinates( self, x, y ):
        # retunrs [xcoord, ycoord] or None if outside
        realCoords = [0,0]
        realCoords[0] = self.theXAxis.convertCoordToNumber( x )

        if realCoords[0] < self.theXAxis.theFrame[0] or realCoords[0] > self.theXAxis.theFrame[1]:
            return None
        realCoords[1] = self.theYAxis.convertCoordToNumber ( y )
        if realCoords[1] < self.theYAxis.theFrame[0] or realCoords[1] > self.theYAxis.theFrame[1]:
            return None
        return realCoords


    def motion(self,obj,event):
        x=event.x
        y=event.y
        self.showTempCoordinates( x, y )
        #if keypressed undo previous  one
        if self.theZoomKeyPressed:
            self.drawxorbox(self.realx0,self.realy0,self.realx1,self.realy1)
            #check whether key is still being pressed, if not cease selection
            state=event.state
            if not (gtk.gdk.BUTTON1_MASK & state):
                self.theZoomKeyPressed=False
            else:
                #get new coordinates, sort them
                #check whether there is excess of boundaries, adjust coordinates
                self.x1=max(self.thePlotArea[0],x)
                self.y1=max(self.thePlotArea[1],y)
                self.x1=min(self.thePlotArea[2]+self.thePlotArea[0],self.x1)
                self.y1=min(self.thePlotArea[3]+self.thePlotArea[1],self.y1)
                #get real coordinates
                self.realx0=min(self.x0,self.x1)
                self.realx1=max(self.x0,self.x1)
                self.realy0=min(self.y0,self.y1)
                self.realy1=max(self.y0,self.y1)
                #draw new rectangle
                self.drawxorbox(self.realx0,self.realy0,self.realx1,self.realy1)


    def release(self,obj,event):
        #check that button 1 is released and previously keypressed was set
        if self.theZoomKeyPressed and event.button==1:
            #draw old inverz rectangle
            self.drawxorbox(self.realx0,self.realy0,self.realx1,self.realy1)

            #call zoomIn    
            self.theZoomKeyPressed=False            
            if self.realx0 < self.realx1 and self.realy0 < self.realy1:
                coords0 = self.convertPlotCoordinates( self.realx0, self.realy0 )
                coords1 = self.convertPlotCoordinates( self.realx1, self.realy1 )

                if coords0 != None and coords1 != None:
                    newxframe = [ coords0[0], coords1[0] ]
                    newyframe = [ coords0[1], coords1[1] ]
                    self.zoomIn(newxframe,newyframe)

        

    def zoomIn(self, newxframe, newyframe):
        #increase zoomlevel
        self.theZoomLevel+=1
        
        #add new frames to zoombuffer
        self.theZoomBuffer.append([self.theXAxis.theFrame[:],
                                self.theYAxis.theFrame[:]])
        self.theXAxis.theFrame[0] = newxframe[0]
        self.theXAxis.theFrame[1] = newxframe[1]
        self.theYAxis.theFrame[0] = newyframe[1]
        self.theYAxis.theFrame[1] = newyframe[0]
        self.requestDataSlice( newxframe[0], newxframe[1] ) 
        self.totalRedraw()
        

    def zoomOut(self):
        #if zoomlevel 0 do nothing
        if self.theZoomLevel == 1:
            self.theZoomLevel -= 1
            self.theZoomBuffer = []
            self.requestData()
        #if zoomlevel 1 delete zoombuffer call setmode(MODE_HISTORY)
        elif self.theZoomLevel > 1:
            self.theZoomLevel -= 1
            newframes = []
            newframes=self.theZoomBuffer.pop()
            self.theXAxis.theFrame = newframes[0]
            self.theYAxis.theFrame = newframes[1]
            self.requestDataSlice( self.theXAxis.theFrame[0],  self.theXAxis.theFrame[1] )
        self.totalRedraw()
            


    def getStripMode(self):
        return self.theStripMode

    def getSeriesCount( self ):
        seriesCount = 0
        for aSeries in self.theSeriesMap.values():
            if aSeries.isOn():
                seriesCount += 1
        return seriesCount


    def printTraceLabels(self):
        #FIXME goes to 2ndary layer
        if self.isControlShown:
            return
        textShift = self.theAscent + self.theDescent + 5
        seriesCount = self.getSeriesCount()
        if seriesCount == 0:
            return
        textShift = min( textShift, int( self.thePlotArea[3]/seriesCount ) )
        y = self.thePlotAreaBox[1] + 5
        x = self.thePlotAreaBox[0] + 5
        
        for aSeries in self.theSeriesMap.values():
            if not aSeries.isOn():
                continue
            self.drawText( BACKGROUND_COLOR, x +1, y, aSeries.getFullPNString() )
            self.drawText( aSeries.getColor(), x, y, aSeries.getFullPNString() )
            y += textShift


    def drawWholePlot(self):

        #clears plotarea
        self.clearPlotarea()
        self.drawFrame()

        #go trace by trace and redraws plot
        for aSeries in self.theSeriesMap.values():
            if aSeries.isOn():
                aSeries.drawAllPoints()
        self. printTraceLabels()


    def drawFrame( self ):
        x0 = self.theOrigo[0]
        y0 = self.theOrigo[1] - self.theYAxis.theLength
        x1 = self.theOrigo[0] + self.theXAxis.theLength
        y1 = self.theOrigo[1]
        self.drawLine(PEN_COLOR, x0, y0, x1, y0 )            
        self.drawLine(PEN_COLOR, x0, y0, x0, y1 )            
        self.drawLine(PEN_COLOR, x1, y0, x1, y1 )            
        self.drawLine(PEN_COLOR, x0, y1, x1, y1 )
        self.drawBox( BACKGROUND_COLOR, self.historyArea[0], self.historyArea[1], self.historyArea[2], self.historyArea[3] )
        self.drawText( PEN_COLOR, self.historyArea[0], self.historyArea[1], self.theStripMode )

    def minimize(self):
        self.theOwner.minimize()           

    
    def maximize(self):
        self.theOwner.maximize()

    def resize( self, newWidth, newHeight ):
        if newWidth < PLOT_MINIMUM_WIDTH:
            newWidth = PLOT_MINIMUM_WIDTH
        if newHeight < PLOT_MINIMUM_HEIGHT:
            newHeight = PLOT_MINIMUM_HEIGHT
        if newWidth == self.thePlotWidth and newHeight == self.thePlotHeight: 
            return
        self.thePlotWidth = newWidth
        self.thePlotHeight  =newHeight
        self.thePixmapBuffer = gtk.gdk.Pixmap( self.theRoot.window, self.thePlotWidth, self.thePlotHeight, -1)
        self.theSecondaryBuffer = gtk.gdk.Pixmap(self.theRoot.window, self.thePlotWidth, self.thePlotHeight, -1)
        self.recalculateSize()
        self.totalRedraw()


# plot display coordinates

