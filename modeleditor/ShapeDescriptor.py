from Constants import *
import Numeric as nu
from gnome.canvas import *

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions(aLabel )
    return tx_width+6, tx_height + 6

class ShapeDescriptor:

    def __init__( self,parentObject, graphUtils, aLabel  ):
        self.theDescriptorList = []
        self.parentObject = parentObject
        self.theGraphUtils = graphUtils
        self.theLabel = aLabel
        # DESCRIPTORLIST SHOULDNT BE DEPENDENT ON WIDTH AND HEIGHT BUT ONLY ON LABEL
        self.thePointMatrix = None  # 8*2 matrix for all points ( (1,RelX,Absx, olw,0,0,0,0),(0,0,0,0,1,Rely,Absy,olw ) )
        # shape name, codename
        # for rect [ point1, point2, point3, point4]
        # for line same
        # for ellipse same
        # for text [point1, point2]
        # FOR BPATH [code, point1,point2...point 6,code...]
        self.theCodeMap = {} 

    def getDescriptorList( self ):
        return self.theDescriptorList

    def getDescriptor ( self, aShapeName ):
        if aShapeName in self.theDescriptorList.keys():
                return self.theDescriptorList[ aShapeName ]
        raise Exception (" Shape %s doesnot exist")


    def getRequiredWidth( self ):
        return 200


    def getRequiredHeight( self ):
        return 20


    def renameLabel (self, newLabel):
        self.theLabel = newLabel
        self.updateShapeDescriptor()

    def getLabel(self):
        return self.theLabel

    def updateShapeDescriptor(self):
        self.reCalculate()

    def getShapeAbsolutePosition(self,aShapeName):
        aDescList=self.getDescriptor(aShapeName)[SD_SPECIFIC][SPEC_POINTS]
        x=aDescList[0] - self.absx
        y=aDescList[1] - self.absy
        return x,y


    def calculateParams( self ):
        propMap =  self.parentObject.thePropertyMap
        if OB_DIMENSION_X in propMap.keys():
            self.width = propMap [ OB_DIMENSION_X ]
        else:
            self.width = 200
        if OB_DIMENSION_Y in propMap.keys():
            self.height = propMap [ OB_DIMENSION_Y ]
        else:
            self.height = 20
        self.absx, self.absy = self.parentObject.getAbsolutePosition()
        (self.tx_height, self.tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel )
        self.tx_height += 2
        self.tx_width += 2
        
        #self.theLabel = self.theGraphUtils.truncateTextToSize( self.theLabel, self.width )
        self.olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )
        

        
    def reCalculate( self ):
        self.calculateParams()
        self.createDescriptorList()

    def createDescriptorList( self ):
        coordsx = nu.array( [[ self.absx], [self.width], [1], [self.olw],[self.tx_width] ] )
        coordsy = nu.array( [[ self.absy], [self.height], [1], [self.olw],[self.tx_height] ] )

        length = len(self.thePointMatrix)
        pointsx= nu.dot( nu.take(self.thePointMatrix,(0,),1), coordsx )
        pointsy= nu.dot( nu.take(self.thePointMatrix,(1,),1), coordsy )
        points = nu.concatenate( (nu.reshape(pointsx,(length,1)), nu.reshape( pointsy,(length,1) ) ),1 )


        for aShapeName in self.theDescriptorList.keys():
            aDescriptor = self.theDescriptorList[ aShapeName ]
            aSpecific = aDescriptor[SD_SPECIFIC]
            aType = aDescriptor[SD_TYPE ]
            aSpecific[0] = []
            if aType in ( CV_RECT, CV_LINE, CV_ELL, CV_TEXT ):
                for aPointCode in self.theCodeMap[ aShapeName]:
                    x=points[aPointCode][0]
                    y=points[aPointCode][1]
                    aSpecific[0].extend( [x,y])
                if aType == CV_TEXT:
                    aSpecific[SPEC_LABEL] = self.theLabel

            elif aType == CV_BPATH:
                for anArtCode in self.theCodeMap[ aShapeName ]:
                    decodedList = []
                    decodedList.append( anArtCode[0] )
                    for aPointCode in anArtCode[1:]:
                        x=points[aPointCode][0]
                        y=points[aPointCode][1]
                        decodedList.extend([x,y])
                    aSpecific[0].append( tuple(decodedList) )
                

"""

class SystemSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        # define pointmatrix
        self.thePointMatrix = nu.array([
                                [[1,0,0,0,0],[1,0,0,0,0]], [[1,1,0,0,0],[1,1,0,0,0]],
                                [[1,0,0,1,0 ], [1,0,0,1,0 ]], [[1,1,0,-1,0 ], [1,0,-1,1,1 ]],
                                [[1,0,0,1,0 ], [1,0,0,1,1 ]], [[1,1,0,-1,0 ], [1,0,0,1,1]],
                                [[1,0,0,1,0 ], [1,0,0,2,1 ]], [[1,1,0,-1,0 ], [1,1,0,-1,0 ]],
                                [[1,0,2,1,0], [1,0,2,0,0]]
                                ])
        self.theCodeMap = {
                    'frame' : [0,1],
                    'textarea' : [2,3],
                    'labelline' : [4,5],
                    'drawarea' : [6,7],
                    'text' : [8]
                    }

        # define descriptorlist
        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
        'frame':['frame', CV_RECT, SD_OUTLINE, SD_OUTLINE, 5, [[ ], 0] ],\
        'textarea':['textarea',CV_RECT, SD_FILL, SD_FILL, 4, [[ ], 0]  ],\
        'labelline':['labelline',CV_LINE,SD_FILL, SD_OUTLINE, 3, [  [], 1 ]], \
        'drawarea':['drawarea', CV_RECT, SD_SYSTEM_CANVAS, SD_FILL, 4, [ [],0 ] ],\
        'text':['text', CV_TEXT, SD_FILL, SD_TEXT, 3, [ [],aLabel ] ] }
        self.reCalculate()


    def getInsideWidth( self ):
        self.calculateParams()
        return self.insideWidth


    def getInsideHeight( self ):
        self.calculateParams()
        return self.insideHeight


    def calculateParams( self ):
        ShapeDescriptor.calculateParams( self )
        self.insideX = self.olw
        self.insideY = self.olw*2+self.tx_height
        self.insideWidth = self.width - self.insideX-self.olw
        self.insideHeight = self.height - self.insideY-self.olw


    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2 + 10


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3 + 10



class ProcessSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        
        self.thePointMatrix = nu.array([\
                                [[1,0,0,0,0],[1,0,0,0,0]], [[1,0,10,0,1],[1,1,0,0,0]],
                                [[1,0,0,2,0 ], [1,0,0,2,0 ]], 
                                [[1,0,5,-1,.5 ], [1,0,0,-1,0 ]], [[1,0,5,1,.5 ], [1,0,0,1,0]],
                                [[1,0,5,-1,0.5 ], [1,1,0,-1,0]], [[1,0,5,1,0.5 ], [1,1,0,1,0 ]],
                                [[1,0,0,-1,0 ], [1,0,5,-1,0.5 ]], [[1,0,0,1,0 ], [1,0,5,1,0.5 ]],
                                [[1,0,10,-1,1 ], [1,0,5,-1,0.5 ]], [[1,0,10,1,1 ], [1,0,5,1,0.5 ]]])
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



    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*3


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3 

    def getRingSize( self ):
        return self.olw*2


class VariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        
        self.thePointMatrix = nu.array([ [[1,0,20,0,0],[1,0,0,0,0]] , #moveto open
                                [[1,1,-20,0,0],[1,0,0,0,0]], #lineto
                                [[1,1,0,0,0],[1,0,0,0,0]],[[1,1,0,0,0],[1,1,0,0,0]],[[1,1,-20,0,0],[1,1,0,0,0]], #curveto
                                [[1,0,20,0,0,],[1,1,0,0,0]],#lineto
                                [[1,0,0,0,0],[1,1,0,0,0]],[[1,0,0,0,0],[1,0,0,0,0]],[[1,0,20,0,0],[1,0,0,0,0]], #curveto
                                [[1,0,20,2,0 ], [1,0,0,2,0 ]], 
                                [[1,0.5,0,-1,0 ], [1,0,0,-1,0 ]], [[1,0.5,0,1,0 ], [1,0,0,1,0]],
                                [[1,0.5,0,-1,0 ], [1,1,0,-1,0 ]], [[1,0.5,0,1,0 ], [1,1,0,1,0 ]],
                                [[1,0,0,0,0 ], [1,0.5,0,-1,0 ]], [[1,0,0,2,0 ], [1,0.5,0,1,0 ]],
                                [[1,1,0,-2,0 ], [1,0.5,0,-1,0 ]], [[1,1,0,0,0 ], [1,0.5,0,1,0 ]]\
                                ])
        self.theCodeMap = {\
                    'frame' : [ [MOVETO_OPEN, 0], [LINETO, 1], [CURVETO, 2,3,4], [LINETO, 5], [CURVETO, 6, 7, 8] ],
                    'text' : [9],
                    RING_TOP : [10,11],
                    RING_BOTTOM : [12,13],
                    RING_LEFT : [14,15],
                    RING_RIGHT : [16,17]    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  

        'frame' : ['frame', CV_BPATH, SD_FILL, SD_FILL, 6, [ [],1 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ [], self.theLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ] ,\
        RING_LEFT : [RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [], 0]  ],\
        RING_RIGHT : [RING_RIGHT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ [],0] ] }
        self.reCalculate()
        



    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2 + 40


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3

    def getRingSize( self ):
        return self.olw*2

class TextSD( ShapeDescriptor ):
    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )
        
        self.thePointMatrix = nu.array([\
                                [[1,0,0,0,0],[1,0,0,0,0]], [[1,1,0,0,0],[1,1,0,0,0]],
                                [[1,0,0,2,0 ], [1,0,0,2,0 ]] \

                                ])
        self.theCodeMap = {\
                    'frame' : [0,1],
                    'text' : [2]\

                    }
        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
        'frame' : ['frame', CV_RECT, SD_FILL, SD_OUTLINE, 6, [ [],0.5 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ [], self.theLabel ] ]}
        self.reCalculate()
        


    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2 + 10 


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3 

        

"""
