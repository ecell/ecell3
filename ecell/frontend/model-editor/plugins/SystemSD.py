#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2012 Keio University
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

import numpy as nu

from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='System' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Default'

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions(aLabel )
    return tx_width+16, tx_height + 20

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


