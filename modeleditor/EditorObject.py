from ComplexShape import *
from Constants import *

class EditorObject:


	def __init__(self, aLayout, objectID,  x, y, parentSystem ):
		self.theLayout = aLayout
		self.theID = objectID
		self.parentSystem = parentSystem
		self.thePropertyMap = {}
		self.thePropertyMap[ OB_POS_X ] = x
		self.thePropertyMap[ OB_POS_Y ] = y
		self.thePropertyMap[ OB_HASFULLID ] = False
		self.theCanvas = None
		self.theShape = None
		# default colors
		self.thePropertyMap [ OB_OUTLINE_COLOR ] = self.theLayout.graphUtils().getRRGByName("black")
		self.thePropertyMap [ OB_FILL_COLOR ] = self.theLayout.graphUtils().getRRGByName("white")
		self.thePropertyMap [ OB_TEXT_COLOR ] = self.theLayout.graphUtils().getRRGByName("blue")
		# default outline
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 1
		self.parentSystem.registerObject( self )



	def destroy(self):
		pass

	def hide( self ):
		# deletes it from canvas
		pass

	def doSelect( self ):
		print "object selected"
		pass

	def showMenu( self ):
		print "object rightclicked"
		pass
	

	def popupEditor( self ):
		print "object doubleclicked"
		self.theLayout.popupObjectEditor( self.theID )		


	def setLimits( self, x0, y0, x1, y1 ):
		pass


	def setCanvas( self, aCanvas ):
		self.theCanvas = aCanvas


	def show( self ):

		self.theShape = ComplexShape( self, self.theCanvas, self.thePropertyMap[ OB_POS_X ], self.thePropertyMap[ OB_POS_Y ], self.thePropertyMap[ OB_DIMENSION_X ], self.thePropertyMap[ OB_DIMENSION_Y ] )
		self.theShape.show()


	def setDetailMode( self, aDetailMode ):
		pass


	def setProperty( self, aPropertyName, aPropertyValue ):
		# if fullID property is changed, change legend, too
		pass


	def getProperty( self, aPropertyName ):
		if aPropertyName in self.thePropertyMap.keys():
			return self.thePropertyMap[aPropertyName]
		else:
			raise Exception("Unknown property %s for object %s"%(self.theName, self.theID ) )


	def getPropertyList( self ):
		pass


	def getAbsolutePosition( self ):
		( xpos, ypos ) = self.parentSystem.getAbsoluteInsidePosition()
		return ( xpos + self.thePropertyMap[ OB_POS_X ], ypos + self.thePropertyMap[ OB_POS_Y ] )


	def getSize( self ):
		pass


	def getRelativePosition( self ):
		pass


	def getPropertyMap( self ):
		pass

	def setPropertyMap( self, aPropertyMap ):
		pass

	def getID( self ):
		return self.theID
		
	def getLayout( self ):
		pass


	def getParent( self ):
		return self.parentObject()

	def getGraphUtils( self ):
		return self.theLayout.graphUtils()

	def getModelEditor( self ):
		return self.theLayout.theLayoutManager.theModelEditor

