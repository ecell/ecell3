from Constants import *
import Numeric as nu
from gnomecanvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Cation'
OB_SHOW_LABEL=1

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    #return tx_width /0.8, 30
    return 30,30

class CationVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        self.thePointMatrix = nu.array([ \
        #frame
        [[1,0,0,0,0],[1,0,0,0,0]],
        #text
        [[1,0,30/2,0,-0.5],[1,1,5,0,0]],
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
                    #frame' : [[MOVETO_OPEN, 0], [CURVETO,1,2,3]],

                    'image' : [0],
                    'text' : [1],
                    RING_TOP : [2,3],
                    RING_BOTTOM : [4,5],
                    RING_LEFT : [6,7],
                    RING_RIGHT : [8,9] 
                    }
        '''
        self.thePointMatrix = nu.array([
        #frame
        [[1,0.45,0,0,0],[1,0.7,0,0,0]], #MOVETO0,0,
        [[1,0.18,0,0,0],[1,0.71,0,0,0 ]],[[1,0.07,0,0,0],[1,0.23,0,0,0 ]],[[1,0.45,0,0,0],[1,0.2,0,0,0 ]],#CURVETO
        [[1,0.79,0,0,0],[1,0.22,0,0,0 ]],[[1,0.76,0,0,0],[1,0.7,0,0,0 ]],[[1,0.45,0,0,0],[1,0.7,0,0,0 ]],#CURVETO
        #text  > [[1,0.1,0,0,0],[1,1.1,0,0,0]]       
        [[1,0,width/2,0,-0.5],[1,0,height+5,0,0]],
        #ring top
        [[1,0,0,-1,0 ],[1,0,0,-1,0 ]],
        [[1,0,0,1,0 ],[1,0,0,1,0 ]], 
        #ring bottom
        [[1,0,0,-1,0 ],[1,1,0,-1,0 ]],
        [[1,0,0,1,0 ],[1,1,0,1,0 ]], 
        #ring left
        [[1,1,0,-1,0 ],[1,1,0,-1,0 ]],
        [[1,1,0,1,0 ],[1,1,0,1,0 ]], 
        #ring right
        [[1,1,0,-1,0 ],[1,0,0,-1,0 ]],
        [[1,1,0,1,0 ],[1,0,0,1,0 ]] ])

        self.theCodeMap = {\
                    'frame' : [[MOVETO_OPEN, 0], [CURVETO,1,2,3], [CURVETO,4,5,6]],
                    'text' : [7],
                    RING_TOP : [8,9],
                    RING_BOTTOM : [10,11],
                    RING_LEFT : [12,13],
                    RING_RIGHT : [14,15] 
                    }
        '''

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'image' : ["image",CV_IMG, SD_FILL, SD_FILL, 3, [ [], "Cation.png" ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ],\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        #(tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        #print str(tx_width + self.olw*2) + ' is estLabelWidth'
        #return tx_width + self.olw*2
        return 30

    def getRequiredWidth( self ):
        #self.calculateParams()
        #print str(self.tx_width + self.olw*2) + ' is getRequiredWidth'
        #return self.tx_width + self.olw*2
        return 30
    '''
    def estLabelDims(graphUtils, aLabel):
        (tx_height, tx_width) = graphUtils.getTextDimensions(aLabel )
        return tx_width+6, tx_height + 6
    '''
    def getRequiredHeight( self ):
        #self.calculateParams()
        #return self.tx_height+ self.olw*4
        return 30
        
    def getRingSize( self ):
        return self.olw*2
        

