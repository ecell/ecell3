from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *


class ProcessObject( EditorObject ):
	
	def __init__( self, aLayout, objectID, aFullID,  x,y, canvas= None ):
		
		EditorObject.__init__( self, aLayout, objectID,x, y, canvas )
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_PROCESS
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_PROCESS
		#default dimensions
		self.theLabel = aFullID.split(':')[2]
		aProcessSD = ProcessSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth

		reqWidth = aProcessSD.getRequiredWidth()
		reqHeight = aProcessSD.getRequiredHeight()
		
	
		self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
		self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight

		
		self.theSD = aProcessSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aProcessSD
		self.thePropertyMap[ PR_CONNECTIONLIST ] = self.getConnectionLineList()

		self.theProcessShapeList=['Rectangle']



	def addConnectionLine( self, aName, aRingNumber, aVariableFullID, endX=None, endY=None ):
		#position by variablefullid takes precedence
		pass


	def deleteConnectionLine( self, aName ):
		pass


	def getConnectionLineList( self ):
		pass



	def reconnect( self, aConnectionName = None):
		pass


	def getAvailableProcessShape(self):
			return self.theProcessShapeList

