
from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

class SystemObject(EditorObject):


	def __init__( self, aLayout, objectID, aFullID,  x, y , parentSystem ):
		EditorObject.__init__( self, aLayout, objectID, x, y, parentSystem )
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_SYSTEM

		#default dimensions
		self.thePropertyMap [ OB_DIMENSION_X ] = 200
		self.thePropertyMap [ OB_DIMENSION_Y ] = 200
		if parentSystem.__class__.__name__ == 'Layout':
			layoutDims = self.theLayout.getProperty( LO_SCROLL_REGION )
			self.thePropertyMap [ OB_DIMENSION_X ] = layoutDims[2] - layoutDims[0]-1
			self.thePropertyMap [ OB_DIMENSION_Y ] = layoutDims[3] - layoutDims[1]-1
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3

		# first get text width and heigth
		self.theLabel = aFullID
		aSystemSD = SystemSD(self, self.getGraphUtils(), self.theLabel )
		self.theSD = aSystemSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aSystemSD
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X  ] = aSystemSD.getInsideWidth()
		self.thePropertyMap[ SY_INSIDE_DIMENSION_Y  ] = aSystemSD.getInsideHeight()


	def registerObject( self, anObject ):
		self.theObjectMap[anObject.getID()] = anObject



	def resize( self , newWidth, newHeigth ):
		pass


	def getEmptyPosition( self ):
		return ( 50,50 )


	def show( self ):
		#render to canvas
		EditorObject.show( self )

	def addItem( self, absx,absy ):
		LE_OBJECT_SYSTEM = 0
		(offsetx, offsety ) = self.getAbsolutePosition()
		x = absx - (self.theSD.insideX + offsetx )
		y = absy - ( self.theSD.insideY + offsety )
		print "absolutre", x, y
		print "relative", x, y
		aSysPath = convertSysIDToSysPath( self.getProperty( OB_FULLID ) )
		if self.theLayout.getPaletteButton() == LE_OBJECT_SYSTEM:
			# create command
			aName = self.getModelEditor().getUniqueEntityName ( ME_SYSTEM_TYPE, aSysPath )
			aFullID = ':'.join( [ME_SYSTEM_TYPE, aSysPath, aName] )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_SYSTEM )
			aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_SYSTEM, aFullID, x, y, self )
			
		self.theLayout.passCommand( [aCommand] )

	def getObjectList( self ):
		# return IDs
		return self.theObjectMap.keys()
		
	def isWithinSystem( self, objectID ):
		#returns true if is within system
		pass
		
	def getAbsoluteInsidePosition( self ):
		( x, y ) = self.getAbsolutePosition()
		return ( x+ self.theSD.insideX, y+self.theSD.insideY )
