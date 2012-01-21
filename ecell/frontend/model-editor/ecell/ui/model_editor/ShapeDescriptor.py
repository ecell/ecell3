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
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas
from ecell.ui.model_editor.Constants import *

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
            if aType in ( CV_RECT, CV_LINE, CV_ELL, CV_TEXT , CV_IMG):
                for aPointCode in self.theCodeMap[ aShapeName]:
                    x=points[aPointCode][0]
                    y=points[aPointCode][1]
                    aSpecific[0].extend( [x,y])
                if aType == CV_TEXT and aShapeName == "text":
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
