from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Process' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Rectangle'
OS_SHOW_LABEL=0

class OtherProcessSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        
        self.thePointMatrix = nu.array([
                                [[1,0,0,0,0],[1,0,0,0,0]], [[1,1,0,0,0],[1,1,0,0,0]],
                                [[1,0,0,2,0 ], [1,0,0,2,0 ]], 
                                [[1,0,5,-1,.5 ], [1,0,0,-1,0 ]], [[1,0,5,1,.5 ], [1,0,0,1,0]],
                                [[1,0,5,-1,0.5 ], [1,1,0,-1,0]], [[1,0,5,1,0.5 ], [1,1,0,1,0 ]],
                                [[1,0,0,-1,0 ], [1,0,5,-1,0.5 ]], [[1,0,0,1,0 ], [1,0,5,1,0.5 ]],
                                [[1,0,10,-1,1 ], [1,0,5,-1,0.5 ]], [[1,0,10,1,1 ], [1,0,5,1,0.5 ]]])
        self.theCodeMap = {\
                    'frame' : [0,1],
                    'text' : [2],
                    RING_TOP : [3,4],
                    RING_BOTTOM : [5,6],
                    RING_LEFT : [7,8],
                    RING_RIGHT : [9,10] }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'frame' : ['frame', CV_RECT, SD_FILL, SD_FILL, 7, [ [], 1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_LEFT : [RING_LEFT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ] }
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        return tx_width + self.olw*3

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*3


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3 

    def getRingSize( self ):
        return self.olw*2

