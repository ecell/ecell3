


class EditorObject:


	def __init__(self, aLayout, objectID, basicType, shapeType, x, y, parentSystem ):
		pass
	

	def destroy(self):
		pass


	def move( self, deltaX, deltaY ):
		pass


	def handleDrag( self, deltaX, deltaY ):
		pass


	def handleMouseClicked( self, x, y ):
		pass

	
	def handleMouseReleased( self ):
		pass
	

	def getMenuItems( self ):
		pass


	def setLimits( self, x0, y0, x1, y1 ):
		pass


	def setCanvas( self, aCanvas ):
		pass


	def show( self ):
		pass


	def hide( self ):
		pass


	def setDetailMode( self, aDetailMode ):
		pass


	def setProperty( self, aPropertyName, aPropertyValue ):
		# if fullID property is changed, change legend, too
		pass


	def getProperty( self, aPropertyName ):
		pass


	def getPropertyList( self ):
		pass


	def getAbsolutePosition( self ):
		pass


	def getSize( self ):
		pass


	def getRelativePosition( self ):
		pass


	def getPropertyMap( self ):
		pass

	def setPropertyMap( self, aPropertyMap ):
		pass
		
	def getID( self ):
		pass
		
	def getLayout( self ):
		pass


	def getParent( self ):
		pass
