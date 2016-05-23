#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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

SHAPE_PLUGIN_TYPE='Process' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Default'
OB_SHOW_LABEL=0

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions(aLabel )
    return 20, 20

class ProcessSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel  )
        # ABS COORD, WIDTH, CONST, OLW, TX_WIDTH
        width =20
        heigth = 15
        self.thePointMatrix = nu.array([ [[1,0,0,0,0],[1,0,0,0,0]],[[1,0,width,0,0 ],[1,0,heigth,0,0]],
                                        [[1,0,width/2,0,-0.5],[1,0,heigth+5,0,0]],
                                        [[1,0,width/2-3,0,0],[1,0,-3,0,0]],[[1,0,width/2+3,0,0],[1,0,3,0,0]],
                                        [[1,0,width/2-3,0,0],[1,0,heigth-3,0,0]],[[1,0,width/2+3,0,0],[1,0,heigth+3,0,0]],
                                        [[1,0,-3,0,0],[1,0,heigth/2-3,0,0]],[[1,0,3,0,0],[1,0,heigth/2+3,0,0]],
                                        [[1,0,width-3,0,0],[1,0,heigth/2-3,0,0]],[[1,0,width+3,0,0],[1,0,heigth/2+3,0,0]]])

        self.theCodeMap = {\
                    'frame' : [0,1],
                    'text'  : [2], 
                    RING_TOP : [3,4],
                    RING_BOTTOM : [5,6],
                    RING_LEFT : [7,8],
                    RING_RIGHT : [9,10] 

                    }

        self.theDescriptorList = {\
        #NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES 
       
        'frame' :[ 'frame', CV_ELL, SD_FILL, SD_FILL, 4, [ [],0 ] ],\
        'text' : ['text', CV_TEXT, SD_FILL, SD_TEXT, 3, [ [], aLabel ] ],\
        RING_TOP : [RING_TOP, CV_RECT, SD_RING, SD_OUTLINE,3, [ [],0 ] ],\
        RING_BOTTOM : [RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [[],0 ] ],\
        RING_LEFT : [RING_LEFT, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ],\
        RING_RIGHT : [RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 3, [ [], 0] ]}
        self.reCalculate()

    def estLabelWidth(self, aLabel):
        return 20

    def getRequiredWidth( self ):
        return 20


    def getRequiredHeight( self ):
        return 15
        
    def getRingSize( self ):
        return 6

    
    

