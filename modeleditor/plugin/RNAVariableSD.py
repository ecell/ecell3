from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='RNA'

class RNAVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        
        self.thePointMatrix = nu.array([ [[1,0,0,0,0,],[1,0,0,0,0]] , #moveto open
                                [[1,1,0,2,0,],[1,-1,0,0,0]], #lineto
                                
                                [[1,1.1,0,2,0.65],[1,0.65,0,0,0]],#lineto
                                [[1,0.45,0,3,0],[1,1.65,0,0,0]],#lineto                                 
[[1,0.45,0,2,0,],[1,0,0,0,0]] , [[1,0.5,0,2,0,],[1,-0.75,0,0,0]],[[1,0.15,0,2,0.5],[1,-0.35,0,0,0]],[[1,0.9,0,2,0,],[1,1.1,0,0,0]],[[1,0.55,0,2,0.5],[1,1.5,0,0,0]],[[1,-0.125,0,2,0.5],[1,0.65,0,0,0]],[[1,0.2,0,3,0],[1,1.1,0,0,0]],[[1,0.825,0,2,0.5],[1,-0.65,0,0,0]],[[1,1.15,0,3,0],[1,-0.25,0,0,0]]                                 ])

        self.theCodeMap = {\
                    'frame' : [ [MOVETO_OPEN, 0], [LINETO, 1],[LINETO, 2],[LINETO, 3],[LINETO,0] ],
                    'text' : [4],
                    RING_TOP : [5,6],
                    RING_BOTTOM : [7,8],
                    RING_LEFT : [9,10],
                    RING_RIGHT : [11,12]    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  

        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 6, [ [],1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ [], self.theLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ],\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [],0] ]}

        self.reCalculate()

    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        return tx_width + self.olw*2

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw
    def getRingSize( self ):
        return self.olw

    
    

