from Constants import *
import Numeric as nu
from gnome.canvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Protein'
#OS_SHOW_LABEL=1

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    return 50, 50

class ProteinVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        self.thePointMatrix = nu.array([ \
        #frame
        [[1,0,0,0,0],[1,0,0,0,0]],
        #text
        [[1,0,50/2,0,-0.5],[1,1,5,0,0]],
        #[[1,0.2,-5,0,0],[1,1,5,0,0]],
        #ring top
        [[1,0.5,0,-1,0 ],[1,-0.1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,-0.1,0,1,0 ]], 
        #ring bottom
        [[1,0.5,0,-1,0 ],[1,1.1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,1.1,0,1,0 ]], 
        #ring left
        [[1,-0.1,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,-0.1,0,1,0 ],[1,0.5,0,1,0 ]], 
        #ring right
        [[1,1.1,0,-1,0 ],[1,0.5,0,-1,0 ]],
        [[1,1.1,0,1,0 ],[1,0.5,0,1,0 ]] ])     
        

        self.theCodeMap = {\
                    'image' : [0],
                    'text' : [1],
                    RING_TOP : [2,3],
                    RING_BOTTOM : [4,5],
                    RING_LEFT : [6,7],
                    RING_RIGHT : [8,9] 
                    }

        '''
        self.thePointMatrix = nu.array([ [[1,0,0,0,0],[1,0.5,0,0,0]] , #MOVETO                                
                                [[1,-0.6,0,0,0],[1,-0.7,0,0,0 ]],[[1,-0.6,0,0,0],[1,-0.7,0,0,0 ]], [[1,0.5,0,0,0],[1,-0.05,0,0,0 ]],#CURVETO L1
                                [[1,1.6,0,0,0],[1,-0.7,0,0,0]],[[1,1.6,0,0,0],[1,-0.7,0,0,0]],[[1,1,0,0,0],[1,0.5,0,0,0]], #CURVETO R2
                                [[1,1.6,0,0,0],[1,1.7,0,0,0 ]],[[1,1.6,0,0,0],[1,1.7,0,0,0 ]], [[1,0.5,0,0,0],[1,1,0,0,0 ]],#CURVETO R3
                                [[1,-0.6,0,0,0],[1,1.7,0,0,0]],[[1,-0.6,0,0,0],[1,1.7,0,0,0]],[[1,0,0,0,0],[1,0.5,0,0,0]], #CURVETO L4
                                #[[1,0.5,0,0,0],[1,3.5,0,0,0]],[[1,0,0,0,0],[1,0.5,0,0,0]],[[1,0,0,0,0],[1,0.5,0,0,0]], #CURVETO L4
        [[1,0,10,2,0 ], [1,0,0,2,0 ]], ### THIS ONE > [[1,0.1,0,0,0],[1,0.15,0,0,0]],
        #[[1,0,0,2,0 ], [1,0,0,2,0 ]],
        #text        
        #[[1,0.1,0,1,0],[1,1.1,0,1   ,0]],
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
                    'frame' : [ [MOVETO_OPEN, 0], [CURVETO, 1,2,3], [CURVETO, 4,5,6], [CURVETO, 7,8,9],[CURVETO, 10,11,12]],
                    'text' : [13],
                    RING_TOP : [14,15],
                    RING_BOTTOM : [16,17],
                    RING_LEFT : [18,19],
                    RING_RIGHT : [20,21]    }
        '''

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
        'image' : ['image',CV_IMG, SD_FILL, SD_FILL, 3, [ [], "Protein.png" ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ [], self.theLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ] ,\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [],0] ] }
        self.reCalculate()
        
    def estLabelWidth(self, aLabel):
        #(tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        #return tx_width + self.olw*2 + 20
        return 50

    def getRequiredWidth( self ):
        #self.calculateParams()
        #return self.tx_width + self.olw*2 + 20
        return 50

    def getRequiredHeight( self ):
        #self.calculateParams()
        #return self.tx_height+ self.olw*3
        return 50

    def getRingSize( self ):
        return self.olw*2



