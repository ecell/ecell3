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

from ecell.ui.model_editor.Utils import *
from ecell.ui.model_editor.EditorObject import *
from ecell.ui.model_editor.Constants import *
from ecell.ui.model_editor.ShapeDescriptor import *
from ecell.ui.model_editor.LayoutCommand import *

class TextObject(EditorObject):


    def __init__( self, aLayout, objectID,  x, y , canvas= None ):
        # text should be in the aFullID argument
        EditorObject.__init__( self, aLayout, objectID, x, y, canvas )
        self.thePropertyMap[ OB_HASFULLID ] = False
        self.theObjectMap = {}
        #self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_TEXT
        self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 1
        self.thePropertyMap[ OB_TYPE ] = OB_TYPE_TEXT
        self.theLabel = 'This is a test string for the text box.'
        aTextSD = TextSD(self, self.getGraphUtils(), self.theLabel )
        # first get text width and heigth

        reqWidth = aTextSD.getRequiredWidth()
        reqHeight = aTextSD.getRequiredHeight()

        self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
        if reqWidth<TEXT_MINWIDTH:
            self.thePropertyMap [ OB_DIMENSION_X ]=TEXT_MINWIDTH

        self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
        if reqHeight<TEXT_MINHEIGHT:
            self.thePropertyMap [ OB_DIMENSION_Y ]=TEXT_MINHEIGHT

        self.theSD = aTextSD
        self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aTextSD
        self.theTextShapeList=['Rectangle']

    def show(self ):
        #render to canvas
        EditorObject.show(self)

    def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
        #first do a resize then a move
        # FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
        # resize must be sum of deltas
        self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
        self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown    


    def reconnect( self ):
        pass

    def getAvailableTextShape(self):
        return self.theTextShapeList


