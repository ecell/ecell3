from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Ion'
OB_SHOW_LABEL=1

class IonVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        
        self.thePointMatrix = nu.array([ [[1,0,0,0,0],[1,0,0,0,0]],[[1,0,4,4,1],[1,0,15,15,1]],[[1,0,4,2,0],[1,0,10,5,0]],[[1,0,4,1.25,0.5],[1,-0.65,0,1.25,0.65]],[[1,0,0,5,0.5],[1,0.25,1,-0.5,-0.15]],[[1,0,4,1.25,0.5],[1,0.45,0,1.25,3.15]],[[1,0,0,5,0.5],[1,1.05,1,0.5,2.65]],[[1,0,1,0.5,0],[1,1,-0.5,1,-0.15]],[[1,0,2,-2,0],[1,-0.1,0.25,9,0.75]],[[1,0,4,5,1],[1,1,-0.5,1,-0.15]],[[1,0,4,2.5,1],[1,-0.1,0.25,9,0.75]]])      

        self.theCodeMap = {\
                    'frame' : [0,1],
                    'text' : [2],
                    RING_TOP : [3,4],
                    RING_BOTTOM : [5,6],
                    RING_LEFT : [7,8],
                    RING_RIGHT : [9,10] 
                    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'frame' : ['frame', CV_ELL, SD_FILL, SD_FILL, 7, [ [], 1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ],\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        return tx_width + self.olw*2

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*4

    def getRingSize( self ):
        return self.olw*2

    

