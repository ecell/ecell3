#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapeDescriptor import *

SHAPE_PLUGIN_TYPE='Variable' #Shape Plugin Constants
SHAPE_PLUGIN_NAME='Default'

def estLabelDims(graphUtils, aLabel):
    (tx_height, tx_width) = graphUtils.getTextDimensions(aLabel )
    return tx_width+46, tx_height + 9



class VariableSD( ShapeDescriptor):

    def __init__( self, parentObject, graphUtils, aLabel ):
        ShapeDescriptor.__init__( self, parentObject, graphUtils, aLabel )

        
        self.thePointMatrix = nu.array([ [[1,0,20,0,0],[1,0,0,0,0]] , #moveto open
                                [[1,1,-20,0,0],[1,0,0,0,0]], #lineto
                                [[1,1,0,0,0],[1,0,0,0,0]],[[1,1,0,0,0],[1,1,0,0,0]],[[1,1,-20,0,0],[1,1,0,0,0]], #curveto
                                [[1,0,20,0,0,],[1,1,0,0,0]],#lineto
                                [[1,0,0,0,0],[1,1,0,0,0]],[[1,0,0,0,0],[1,0,0,0,0]],[[1,0,20,0,0],[1,0,0,0,0]], #curveto
                                [[1,0,20,2,0 ], [1,0,0,2,0 ]], #text
                                [[1,0.5,0,-1,0 ], [1,0,0,-1,0 ]], [[1,0.5,0,1,0 ], [1,0,0,1,0]],
                                [[1,0.5,0,-1,0 ], [1,1,0,-1,0 ]], [[1,0.5,0,1,0 ], [1,1,0,1,0 ]],
                                [[1,0,0,0,0 ], [1,0.5,0,-1,0 ]], [[1,0,0,2,0 ], [1,0.5,0,1,0 ]],
                                [[1,1,0,-2,0 ], [1,0.5,0,-1,0 ]], [[1,1,0,0,0 ], [1,0.5,0,1,0 ]]\
                                ])
        self.theCodeMap = {\
                    'frame' : [ [ gnomecanvas.MOVETO_OPEN, 0], [ gnomecanvas.LINETO, 1], [gnomecanvas.CURVETO, 2,3,4], [gnomecanvas.LINETO, 5], [ gnomecanvas.CURVETO, 6, 7, 8] ],
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
        
    def estLabelWidth(self, aLabel):
        (tx_height, tx_width) = self.theGraphUtils.getTextDimensions(aLabel )
        return tx_width+self.olw*2+40


    def getRequiredWidth( self ):
        self.calculateParams()
        return self.tx_width + self.olw*2 + 40


    def getRequiredHeight( self ):
        self.calculateParams()
        return self.tx_height+ self.olw*3

    def getRingSize( self ):
        return self.olw*2
"""
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
