from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

class VariableObject( EditorObject ):
	
	def __init__( self, aLayout,objectID, aFullID,  x,y, canvas= None ):
		EditorObject.__init__( self, aLayout, objectID, x, y, canvas)
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_VARIABLE
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_VARIABLE
		#default dimensions
		self.theLabel = aFullID
		aVariableSD = VariableSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth

		reqWidth = aVariableSD.getRequiredWidth()
		reqHeight = aVariableSD.getRequiredHeight()

		self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
		self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight


		self.theVD = aVariableSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aVariableSD
		self.theVariableShapeList=['Rounded Rectangle']



	def reconnect( self ):
		pass


	def getAvailableVariableShape(self):
		return self.theVariableShapeList

