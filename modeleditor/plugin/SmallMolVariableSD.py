from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='SmallMol'
OS_SHOW_LABEL=1
def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    return tx_width +9, 30
class SmallMolVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        self.thePointMatrix = nu.array([
        #Frame 
        [[1,0.26,0,0,0],[1,0.23,0,0,0]], #MOVETO
        [[1,0.30,0,0,0],[1,-0.6,0,0,0 ]],[[1,1.3,0,0,0],[1,0.35,0,0,0 ]],[[1,0.76,0,0,0],[1,0.72,0,0,0 ]],#CURVETO
        [[1,0.51,0,0,0],[1,1.8,0,0,0 ]],[[1,-0.4,0,0,0],[1,0.6,0,0,0 ]],[[1,0.26,0,0,0],[1,0.23,0,0,0 ]],#CURVETO
        #text        
        [[1,0.1,0,0,0],[1,1.1,0,0,0]], #0.1, 1.1        
        #ring top
        [[1,0.5,0,-1,0 ],[1,-0.1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,-0.1,0,1,0 ]], 
        #ring bottom
        [[1,0.5,0,-1,0 ],[1,1.1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,1.1,0,1,0 ]], 
        #ring left
        [[1,0.05,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,0.05,0,1,0 ],[1,0.5,0,1,0 ]], 

        #ring right
        [[1,0.9,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,0.9,0,1,0 ],[1,0.5,0,1,0 ]] ]) 
        ''' SmallMol Zj
        #Ring Top
        [[1,0,4,1.25,0.5],[1,-0.65,0,1.25,0.65]],
        [[1,0,0,5,0.5],[1,0.25,1,-0.5,-0.15]],
        #Ring Bottom
        [[1,0,4,1.25,0.5],[1,0.45,0,1.25,0.65]],
        [[1,0,0,5,0.5],[1,1.05,1,0.5,0.15]],
        #Ring Left
        [[1,0,1,0.5,0],[1,1,-0.5,1,-0.5]],
        [[1,0,2,-2,0],[1,-0.1,0.25,9,-0.5]],
        #Ring Right
        [[1,0,4,5,1],[1,1,-0.5,1,-0.5]],
        [[1,0,4,2.5,1],[1,-0.1,0.25,9,-0.5]]])      
        '''

        self.theCodeMap = {\
                    'frame' : [[MOVETO_OPEN, 0], [CURVETO, 1,2,3], [CURVETO, 4,5,6]],
                    'text' : [7],
                    RING_TOP : [8,9],
                    RING_BOTTOM : [10,11],
                    RING_LEFT : [12,13],
                    RING_RIGHT : [14,15]
                    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 7, [ [], 1 ] ],\
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

    

