from Constants import *
import numpy as nu
from gnomecanvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='SmallMol'
OS_SHOW_LABEL=1

def estLabelDims(graphUtils, aLabel):
    #(tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    #return tx_width /0.8, 30
    return 30,30

class SmallMolVariableSD( ShapeDescriptor):

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


        '''
        self.thePointMatrix = nu.array([
        #Frame 
        [[1,0.25,0,0,0],[1,0.32,0,0,0]], #MOVETO
        [[1,0.32,0,0,0],[1,-0.4,0,0,0 ]],[[1,1.5,0,0,0],[1,0.34,0,0,0 ]],[[1,0.72,0,0,0],[1,0.69,0,0,0 ]],#CURVETO
        [[1,0.6,0,0,0],[1,1.39,0,0,0 ]],[[1,-0.48,0,0,0],[1,0.58,0,0,0 ]],[[1,0.25,0,0,0],[1,0.32,0,0,0 ]],#CURVETO
        #text > [[1,0.1,0,0,0],[1,1.1,0,0,0]]
        [[1,0,width/2,0,-0.5],[1,-1.1,height+5,0,0]], #0.1, 1.1        
        #ring top
        [[1,0.5,0,-1,0 ],[1,0,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,0,0,1,0 ]], 
        #ring bottom
        [[1,0.5,0,-1,0 ],[1,1,0,-1,0 ]],
        [[1,0.5,0,1,0 ],[1,1,0,1,0 ]], 
        #ring left
        [[1,0,0,-1,0 ],[1,0.6,0,-1,0 ]],
        [[1,0,0,1,0 ],[1,0.6,0,1,0 ]], 
        #ring right
        [[1,1,0,-1,0 ],[1,0.4,0,-1,0 ]],
        [[1,1,0,1,0 ],[1,0.4,0,1,0 ]] ])       
        '''
        '''
        self.theCodeMap = {\
                    'frame' : [[MOVETO_OPEN, 0], [CURVETO, 1,2,3], [CURVETO, 4,5,6]],
                    'text' : [7],
                    RING_TOP : [8,9],
                    RING_BOTTOM : [10,11],
                    RING_LEFT : [12,13],
                    RING_RIGHT : [14,15]
                    }
        '''

        self.theCodeMap = {\
                    #frame' : [[MOVETO_OPEN, 0], [CURVETO,1,2,3]],
                    'image' : [0],
                    'text' : [1],
                    RING_TOP : [2,3],
                    RING_BOTTOM : [4,5],
                    RING_LEFT : [6,7],
                    RING_RIGHT : [8,9] 
                    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'image' : ["image",CV_IMG, SD_FILL, SD_FILL, 3, [ [], "SmallMol.png" ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ],\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        #(tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        #return tx_width / 0.8
        return 30

    def getRequiredWidth( self ):
        #self.calculateParams()
        #return self.tx_width /0.8
        return 30

    def getRequiredHeight( self ):
        #self.calculateParams()
        return 30

    def getRingSize( self ):
        return self.olw*2

    

