
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
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3

		#default dimensions
		self.thePropertyMap [ OB_DIMENSION_X ] = 200
		self.thePropertyMap [ OB_DIMENSION_Y ] = 200
		self.theLabel = aFullID
		aSystemSD = SystemSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth

		reqWidth = aSystemSD.getRequiredWidth()
		reqHeight = aSystemSD.getRequiredHeight()

		if parentSystem.__class__.__name__ == 'Layout':
			layoutDims = self.theLayout.getProperty( LO_SCROLL_REGION )
			self.thePropertyMap [ OB_DIMENSION_X ] = layoutDims[2] - layoutDims[0]-1
			self.thePropertyMap [ OB_DIMENSION_Y ] = layoutDims[3] - layoutDims[1]-1
		else:
			if reqWidth > self.thePropertyMap [ OB_DIMENSION_X ]:
				self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
			if reqHeight > self.thePropertyMap [ OB_DIMENSION_Y ]:
				self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
			spaceleftX = self.parentSystem.getProperty( SY_INSIDE_DIMENSION_X ) - self.getProperty( OB_DIMENSION_X ) - self.getProperty( OB_POS_X )
			spaceleftY = self.parentSystem.getProperty( SY_INSIDE_DIMENSION_Y ) - self.getProperty( OB_DIMENSION_Y ) - self.getProperty( OB_POS_Y )
			spaceleft = min( spaceleftX, spaceleftY )/2 
			if spaceleft > 10:
				self.thePropertyMap [ OB_DIMENSION_Y ] += spaceleft 
				self.thePropertyMap [ OB_DIMENSION_X ] += spaceleft 


		self.theSD = aSystemSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aSystemSD
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X  ] = aSystemSD.getInsideWidth()
		self.thePropertyMap[ SY_INSIDE_DIMENSION_Y  ] = aSystemSD.getInsideHeight()


	def registerObject( self, anObject ):
		self.theObjectMap[anObject.getID()] = anObject



	def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
		#first do a resize then a move
		# FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
		# resize must be sum of deltas
		self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
		self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown 
		self.theShape.resize( deltaleft + deltaright, deltaup + deltadown )
		self.thePropertyMap[ OB_DIMENSION_X ]
		self.thePropertyMap[ OB_DIMENSION_X ]
		self.move( -deltaleft, -deltaup )



	def getEmptyPosition( self ):
		return ( 50,50 )


	def show( self ):
		#render to canvas
		EditorObject.show( self )

	def addItem( self, absx,absy ):

		(offsetx, offsety ) = self.getAbsolutePosition()
		x = absx - (self.theSD.insideX + offsetx )
		y = absy - ( self.theSD.insideY + offsety )
		aSysPath = convertSysIDToSysPath( self.getProperty( OB_FULLID ) )
		aCommand = None
		buttonPressed = self.theLayout.getPaletteButton()
		if  buttonPressed == PE_SYSTEM:
			# create command
			aName = self.getModelEditor().getUniqueEntityName ( ME_SYSTEM_TYPE, aSysPath )
			aFullID = ':'.join( [ME_SYSTEM_TYPE, aSysPath, aName] )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_SYSTEM )
			aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_SYSTEM, aFullID, x, y, self )
		elif buttonPressed == PE_PROCESS:
			print "process button pressed"
		elif buttonPressed == PE_VARIABLE:
			print "variable button pressed"
		elif buttonPressed == PE_TEXT:
			print "text button pressed"
		elif buttonPressed == PE_SELECTOR:
			self.doSelect()
		elif buttonPressed == PE_CUSTOM:
			print "custom button pressed"


		if aCommand != None:
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


	def outlineDragged( self, deltax, deltay, absx, absy ):

		olw = self.getProperty( OB_OUTLINE_WIDTH )
		width = self.getProperty( OB_DIMENSION_X )
		height = self.getProperty( OB_DIMENSION_Y )
		(offsetx, offsety ) = self.getAbsolutePosition()
		x = absx- offsetx
		y = absy - offsety

		direction = 0
		deltaup = 0
		deltadown = 0
		deltaleft = 0
		deltaright = 0
		#upwards direction:
		if x <= olw:
			direction |= DIRECTION_UP
			deltaleft = -deltax

		# downwards direction
		elif x>= width -olw:
			direction |= DIRECTION_DOWN
			deltaright = deltax


		# leftwise direction
		if y <= olw:
			direction |= DIRECTION_LEFT
			deltaup = - deltay

		# rightwise direction
		elif y>= height - olw:
			direction |= DIRECTION_RIGHT
			deltadown = deltay

		if direction != 0:
			#FIXMEparentSystem boundaries should be watched!!!

			aCommand = ResizeObject( self.theLayout, self.theID, deltaup, deltadown, deltaleft, deltaright )
			self.theLayout.passCommand( [aCommand] )
