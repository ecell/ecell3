from Constants import *
import numpy as nu
from gnomecanvas import *
from ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Process' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Rectangle'
OS_SHOW_LABEL=0

class OtherProcessSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        
        self.thePointMatrix = nu.array([
                                #frame
                                [[1,0,0,0,0],[1,0,0,0,0]], 
                                [[1,1,0,0,0],[1,1,0,0,0]],
                                #text
                                [[1,0,10,2,0 ], [1,-0.35,0,2,0 ]], 
                                # THIS ONE > [[1,0.1,0,0,0],[1,0.1,0,0,0]],
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
        '''
                                #text
                                [[1,-1.5,0,2,0 ], [1,1,0,2,0 ]],
                                #Ring Top 
                                [[1,-1.65,5,-1,.5 ], [1,-.03,0,-1,0 ]], 
                                [[1,-1.65,5,1,.5 ], [1,-.03,0,1,0]],
                                #Ring Bottom
                                [[1,-1.65,5,-1,0.5 ], [1,1,0,-1,0]], 
                                [[1,-1.65,5,1,0.5 ], [1,1,0,1,0 ]],
                                #Ring Left
                                [[1,0,0,-1,0 ], [1,-.4,5,-1,0.5 ]], 
                                [[1,0,0,1,0 ], [1,-.4,5,1,0.5 ]],
                                #Ring Right
                                [[1,-3.3,10,-1,1 ], [1,-.4,5,-1,0.5 ]], 
                                [[1,-3.3,10,1,1 ], [1,-.4,5,1,0.5 ]]])
        '''
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
        return tx_width + self.olw*3 + 20

    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*3 + 20


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3 

    def getRingSize( self ):
        return self.olw*2

