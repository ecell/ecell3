from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

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


