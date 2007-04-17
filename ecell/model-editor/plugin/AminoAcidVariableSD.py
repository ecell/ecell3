from Constants import *
import numpy as nu
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='AminoAcid'
OS_SHOW_LABEL=1

def estLabelDims(graphUtils, aLabel):
    #(tx_height, tx_width) = graphUtils.getTextDimensions( aLabel )
    return 40, 40 

class AminoAcidVariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        #[[ self.absx], [self.width], [1], [self.olw],[self.tx_width] ]

        '''
        #frame
        [[1,0.01,0,0,0],[1,0.98,0,0,0]], #MOVETO
        [[1,0.50,0,0,0],[1,0.50,0,0,0 ]], #LINETO
        [[1,0.50,0,0,0],[1,0.02,0,0,0 ]], #LINETO
        [[1,0.50,0,0,0],[1,0.50,0,0,0 ]], #LINETO
        [[1,0.98,0,0,0],[1,0.98,0,0,0 ]], #LINETO
        [[1,0.50,0,0,0],[1,0.50,0,0,0 ]], #LINETO
        [[1,0.01,0,0,0],[1,0.98,0,0,0 ]], #LINETO
        #text COOH        
        [[1,0.5,0,0,0],[1,0.6,0,0,0]],
        #text NH2       
        [[1,1,0,0,0],[1,0.6,0,0,0]],
        #text > [[1,0.1,0,0,0],[1,1.1,0,0,0]]       
        '''        
        
        self.thePointMatrix = nu.array([ \
        #frame
        [[1,0,0,0,0],[1,0,0,0,0]],
        #text
        [[1,0,40/2,0,-0.5],[1,1,5,0,0]],
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
        'frame' : [[MOVETO_OPEN, 0], [LINETO,1], [LINETO,2],[LINETO,3],[LINETO,4,],[LINETO,5],[LINETO,6]],
        'freetext' : [7],
        'freetext' : [8],
        '''

        self.theCodeMap = {\
                    'image' : [0],
                    'text' : [1],
                    RING_TOP : [2,3],
                    RING_BOTTOM : [4,5],
                    RING_LEFT : [6,7],
                    RING_RIGHT : [8,9] 
                    }

        '''
        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 7, [ [], 1 ] ],\
        'freetext' : ['freetext', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], 'COOH' ] ],\
        'freetext' : ['freetext', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], 'NH2' ] ],\
        '''

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
        'image' : ['image',CV_IMG, SD_FILL, SD_FILL, 3, [ [], "Amino.png" ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 5, [[],0 ] ],\
        RING_LEFT : [RING_LEFT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        #(tx_height, tx_width) = self.theGraphUtils.getTextDimensions( aLabel )
        #return tx_width / 0.8 + 40
        return 40

    def getRequiredWidth( self ):
        #self.calculateParams()
        #return self.tx_width /0.8 + 40
        return 40

    def getRequiredHeight( self ):
        #self.calculateParams()
        return 40

    def getRingSize( self ):
        return self.olw*2   




