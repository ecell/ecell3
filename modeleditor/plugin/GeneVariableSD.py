from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Gene'
OS_SHOW_LABEL=1
def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    return tx_width /0.8, 30

class GeneVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        #[[ self.absx], [self.width], [1], [self.olw],[self.tx_width] ]
        self.thePointMatrix = nu.array([ \
        #frame
        [[1,0,0,0,0],[1,0,0,0,0]], #MOVETO0,0,
        [[1,0.2,0,0,0],[1,0.1,0,0,0 ]], #LINETO  0.2, 0.1
        [[1,0.7,0,0,0],[1,1.9,0,0,0 ]],[[1,0.7,0,0,0],[1,-0.8,0,0,0 ]],[[1,1,0,0,0],[1,1,0,0,0 ]],#CURVETO 0.7,1.9, 0.7,-0.8, 1.0, 1.0
        [[1,0.8,0,0,0],[1,0.9,0,0,0 ]],#LINETO 0.8,0.9
        [[1,0.5,0,0,0],[1,-0.9,0,0,0 ]],[[1,0.5,0,0,0],[1,1.8,0,0,0 ]],[[1,0,0,0,0],[1,0,0,0,0 ]],#CURVETO 0.5,  -0.9, 0.5, 1.8, 0,0
        
        #text        
        [[1,0.1,0,0,0],[1,1.1,0,0,0]], #0.1, 1.1
        
        #ring top
        [[1,0.5,0,-1,0 ],[1,0,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,0,0,1,0 ]], 

        #ring bottom
        [[1,0.5,0,-1,0 ],[1,1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,1,0,1,0 ]], 

        #ring left
        [[1,0,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,0,0,1,0 ],[1,0.5,0,1,0 ]], 

        #ring right
        [[1,1,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,1,0,1,0 ],[1,0.5,0,1,0 ]] ])      

        self.theCodeMap = {\
                    'frame' : [[MOVETO_OPEN, 0], [LINETO,1], [CURVETO,2,3,4], [LINETO,5], [CURVETO,6,7,8] ],
                    'text' : [9],
                    RING_TOP : [10,11],
                    RING_BOTTOM : [12,13],
                    RING_LEFT : [14,15],
                    RING_RIGHT : [16,17] 
                    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 7, [ [], 1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_LEFT : [RING_LEFT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        return tx_width / 0.8

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width /0.8


    def getRequiredHeight( self ):
        #self.calculateParams()
        return 30

    def getRingSize( self ):
        return self.olw*2

    

