from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

class TextObject(EditorObject):


	def __init__( self, aLayout, objectID, aFullID,  x, y , canvas= None ):
		EditorObject.__init__( self, aLayout, objectID, x, y, canvas )
		print 'TextObject init'
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		#self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_TEXT
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 1
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_TEXT
		self.theLabel = aFullID
		aTextSD = TextSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth

		reqWidth = aTextSD.getRequiredWidth()
		reqHeight = aTextSD.getRequiredHeight()

		self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
		self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight


		self.theSD = aTextSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aTextSD

	def show(self ):
		#render to canvas
		EditorObject.show(self)


	def reconnect( self ):
		pass

	def getAvailableVariableShape(self):
		return self.theVariableShapeList


