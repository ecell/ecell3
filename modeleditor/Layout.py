
from LayoutManager import *
from ModelStore import *
from PackingStrategy import *

class Layout:

	def __init__( self, aLayoutManager, aName ):
		self.theLayoutManager = aLayoutManager
		self.theLayoutBufferFactory = self.theLayoutManager.theLayoutBufferFactory
		self.thePackingStrategy = PackingStrategy( self )
		self.theLayoutBufferPaster = self.theLayoutManager.theLayoutBufferPaster
		self.theName = aName
		self.theObjectMap = {}
		self.theCanvas = None

	def update( self, aType = None, anID = None ):
		# i am not sure this is necessary
		pass

	def attachToCanvas( self, aCanvas ):
		self.theCanvas = aCanvas
		# set canvas scroll region
		scrollRegion = self.getProperty( LO_SCROLL_REGION )
		self.theCanvas.set_scroll_region( scrollRegion[0], scrollRegion[1], scrollRegion[2], scrollRegion[3])
		# set canvas ppu
		ppu = self.getProperty( LO_PIXELS_PER_UNIT )
		self.theCanvas.set_pixels_per_unit( ppu )
		
		# set canvas for objects and show objects
		for objectID in self.theObjectMap.keys():
			anObject = self.theObjectMap[ objectID ]
			anObject.setCanvas( self.theCanvas )
			anObject.show()
		


	def detachFromCanvas( self ):
		# hide objects and setcanvas none
		for objectID in self.theObjectMap.keys():
			anObject = self.theObjectMap[ objectID ]
			anObject.setCanvas( None )
			anObject.hide()
		
		self.theCanvas = None
		


	def getCanvas( self ):
		return self.theCanvas
		

	def rename( self, newName ):
		self.theName = newName
		self.theCanvas.getParentWindow.update()
		
		
	#########################################
	#         	COMMANDS		#
	#########################################


	def createObject( self, objectID, objectType, aFullID, x=None, y=None, parentSystem  ):
		# object must be within a system except for textboxes 
		pass


	def deleteObject( self, anObjectID ):
		pass


	def getObjectList( self, anObjectType = None ):
		# returns IDs
		pass


	def getPropertyList( self ):
		pass
		
	
	def getProperty( self ):
		pass
	
	
	def setProperty( self, aPropertyName, aValue ):
		pass
	

	def moveObject(self, anObjectID, newX, newY, newParent ):
		# if newParent is None, means same system
		pass


	def getObject( self, anObjectID ):
		# returns the object including connectionobject
		pass


	def resizeObject( self, anObjectID, deltaTop, deltaBottom, deltaLeft, deltaRight ):
		# inward movement negative, outward positive
		pass


	def createConnectionObject( self, anObjectID, aProcessObjectID = None, aVariableObjectID=None,  processRing=None, variableRing=None, direction = PROCESS_TO_VARIABLE, aVarrefName ):
		# if processobjectid or variableobjectid is None -> no change on their part
		# if process or variableID is the same as connection objectid, means that it should be left unattached
		pass


	def redirectConnectionObject( self, anObjectID, newProcessObjectID, newVariableObjectID = None, processRing = None, variableRing = None ):
		# if processobjectid or variableobjectid is None -> no change on their part
		# if process or variableID is the same as connection objectid, means that it should be left unattached
		pass


	#################################################
	#		USER INTERACTIONS		#
	#################################################

	def userMoveObject( self, ObjectID, deltaX, deltaY ):
		#to be called after user releases shape
		pass
		#return TRUE move accepted, FALSE move rejected


	def userCreateConnection( self, aProcessObjectID, startRing, targetx, targety ):
		pass
		#return TRUE if line accepted, FALSE if line rejected


	def autoAddSystem( self, aFullID ):
		pass


	def autoAddEntity( self, aFullID ):
		pass


	def autoConnect( self, aProcessFullID, aVariableFullID, aName ):
		pass



	####################################################
	# 			OTHER			   #
	####################################################

	def getUniqueObjectID( self, anObjectType ):
		# objectID should be string
		pass


	def getName( self ):
		return self.theName
		
