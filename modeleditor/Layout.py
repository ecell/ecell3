from Constants import *
from LayoutManager import *
from ModelStore import *
from PackingStrategy import *
from SystemObject import *
import gnome.canvas 

class Layout:

	def __init__( self, aLayoutManager, aName ):
		self.theLayoutManager = aLayoutManager
		self.theLayoutBufferFactory = self.theLayoutManager.theLayoutBufferFactory
		self.thePackingStrategy = PackingStrategy( self )
		self.theLayoutBufferPaster = self.theLayoutManager.theLayoutBufferPaster
		self.theName = aName
		self.theObjectMap = {}
		self.thePropertyMap = {}
		default_scrollregion = [ -1000, -1000, 1000, 1000 ]
		default_zoomratio = 1
		self.thePropertyMap[ LO_SCROLL_REGION ] = default_scrollregion
		self.thePropertyMap[ LO_ZOOM_RATIO ] = default_zoomratio
		self.theCanvas = None
		self.thePathwayEditor = None

		# allways add root dir object
		anObjectID = self.getUniqueObjectID( ME_SYSTEM_TYPE )
		self.createObject( anObjectID, ME_SYSTEM_TYPE, ME_ROOTID, default_scrollregion[0], default_scrollregion[1], None )


	def update( self, aType = None, anID = None ):
		# i am not sure this is necessary
		pass

	def attachToCanvas( self, aCanvas ):
		self.theCanvas = aCanvas
		self.thePathwayEditor = self.theCanvas.getParentWindow()
		self.theCanvas.setLayout( self )
		# set canvas scroll region
		scrollRegion = self.getProperty( LO_SCROLL_REGION )
		self.theCanvas.setSize( scrollRegion )
		# set canvas ppu
		ppu = self.getProperty( LO_ZOOM_RATIO )
		self.theCanvas.setZoomRatio( ppu )

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


	def createObject( self, objectID, objectType, aFullID, x=None, y=None, parentSystem = None  ):
		# object must be within a system except for textboxes 
		# parentSystem object cannot be None, just for root
		if x == None and y == None:
			(x,y) = parentSystem.getEmptyPosition()

		if objectType == OB_TYPE_PROCESS:
			pass
		elif objectType == OB_TYPE_VARIABLE:
			pass
		elif objectType == OB_TYPE_SYSTEM:
			if parentSystem == None:
				parentSystem = self
			newObject = SystemObject( self, objectID, aFullID, x, y, parentSystem )

		elif objectType == OB_TYPE_TEXT:
			pass
		elif objectType == OB_TYPE_CONNECTION:
			pass
		else:
			raise Exception("Object type %s does not exists"%objectType)
		self.theObjectMap[ objectID ] = newObject
		if self.theCanvas!=None:
			newObject.setCanvas( self.theCanvas )
			newObject.show()


	def deleteObject( self, anObjectID ):
		pass


	def getObjectList( self, anObjectType = None ):
		# returns IDs
		return self.theObjectMap.keys()


	def getPropertyList( self ):
		return self.thePropertyMap.keys()
		
	
	def getProperty( self, aPropertyName ):
		if aPropertyName in self.thePropertyMap.keys():
			return self.thePropertyMap[aPropertyName]
		else:
			raise Exception("Unknown property %s for layout %s"%(self.theName, self.thePropertyName ) )
	
	
	def setProperty( self, aPropertyName, aValue ):
		pass

	def getAbsoluteInsidePosition( self ):
		return ( 0, 0 )

	def moveObject(self, anObjectID, newX, newY, newParent ):
		# if newParent is None, means same system
		pass


	def getObject( self, anObjectID ):
		# returns the object including connectionobject
		if anObjectID not in self.theObjectMap.keys():
			raise Exception("%s objectid not in layout %s"%(anObjectID, self.theName))
		return self.theObjectMap[ anObjectID ]


	def resizeObject( self, anObjectID, deltaTop, deltaBottom, deltaLeft, deltaRight ):
		# inward movement negative, outward positive
		pass


	def createConnectionObject( self, anObjectID, aProcessObjectID = None, aVariableObjectID=None,  processRing=None, variableRing=None, direction = PROCESS_TO_VARIABLE, aVarrefName = None ):
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
		counter = 0
		while anObjectType + str( counter) in self.theObjectMap.keys():
			counter += 1
		return anObjectType + str( counter )


	def getName( self ):
		return self.theName
		

	def graphUtils( self ):
		return self.theLayoutManager.theGraphicalUtils


	def popupObjectEditor( self, anObjectID ):
		pass

	def getPaletteButton( self ):
		LE_OBJECT_SYSTEM = 0
		return LE_OBJECT_SYSTEM

	def passCommand( self, aCommandList):
		self.theLayoutManager.theModelEditor.doCommandList( aCommandList)

	def registerObject( self, anObject ):
		self.theRootObject = anObject
