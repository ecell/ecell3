from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Protein'
OS_SHOW_LABEL=1
def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    return tx_width /0.7, 30
    
class ProteinVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )
       
        self.thePointMatrix = nu.array([ 
        [[1,0.15,0,0,0],[1,0.06,0,0,0]], #MOVETO0,0,
        [[1,0.325,0,0,0],[1,-0.07,0,0,0 ]],[[1,0.24,0,0,0],[1,0.32,0,0,0 ]],[[1,0.71,0,0,0],[1,0.05,0,0,0 ]],#CURVETO 0.7,1.9, 0.7,-0.8, 1.0, 1.0
        [[1,0.9,0,0,0],[1,-0.05,0,0,0 ]],[[1,1.12,0,0,0],[1,0.5,0,0,0 ]],[[1,0.92,0,0,0],[1,0.8,0,0,0 ]],#CURVETO 0.7,1.9, 0.7,-0.8, 1.0, 1.0
        [[1,0.5,0,0,0],[1,1.2,0,0,0 ]],[[1,0.53,0,0,0],[1,0.78,0,0,0 ]],[[1,0.3,0,0,0],[1,0.81,0,0,0 ]],#CURVETO 0.7,1.9, 0.7,-0.8, 1.0, 1.0
        [[1,0.03,0,0,0],[1,0.89,0,0,0 ]],[[1,-0.12,0,0,0],[1,0.22,0,0,0 ]],[[1,0.15,0,0,0],[1,0.06,0,0,0 ]],#CURVETO 0.7,1.9, 0.7,-0.8, 1.0, 1.0
        
        
        #text        
        [[1,0.15,0,0,0],[1,0.1,0,0,0]], #0.1, 1.1
        
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
                    'frame' : [ [MOVETO_OPEN, 0], [CURVETO, 1,2,3], [CURVETO, 4,5,6], [CURVETO, 7, 8, 9], [CURVETO, 10, 11, 12] ],
                    'text' : [13],
                    RING_TOP : [14,15],
                    RING_BOTTOM : [16,17],
                    RING_LEFT : [18,19],
                    RING_RIGHT : [20,21]    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  

        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 6, [ [],1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ [], self.theLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ] ,\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [],0] ] }
        self.reCalculate()
        
    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        return tx_width /0.7 

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width /0.7


    def getRequiredHeight( self ):
        self.calculateParams()
        return 30

    def getRingSize( self ):
        return self.olw*2

