from Buffer import *


class ObjectBuffer:

	def __init__( self, anID ):
		self.theID = anID
		self.thePropertyBuffer = PropertyListBuffer()
		self.theEntityBuffer = None
		self.theParent = None
		self.undoFlag = False

	def getID( self ):
		return self.theID

	def getPropertyList( self ):
		return self.thePropertyBuffer.getPropertyList()

	def getProperty( self, aPropertyName ):
		return self.thePropertyBuffer.getProperty( aPropertyName )

	def getPropertyBuffer ( self ):
		return self.thePropertyBuffer

	def getEntityBuffer( self ):
		if self.theEntityBuffer == None:
			raise Exception( "Buffer of object %s doesn't have entitybuffer"%self.theID )
		return self.theEntityBuffer
		
	def setEntityBuffer( self, anEntityBuffer ):
		self.theEntityBuffer = anEntityBuffer


class ObjectListBuffer:

	def __init__( self ):
		self.theObjectList = {}


	def getObjectBufferList( self ):
		return self.theObjectList.keys()


	def getObjectBuffer( self, objectID ):
		return self.theObjectList[ objectID ]


	def addObjectBuffer( self, objectBuffer ):
		self.theObjectList[ objectBuffer.getID() ] = objectBuffer



class SystemObjectBuffer(ObjectBuffer):

	def __init__( self, anID ):
		ObjectBuffer.__init__( self, anID )
		self.theSingleObjectListBuffer = ObjectListBuffer()
		self.theSystemObjectListBuffer = ObjectListBuffer()

	def getSingleObjectListBuffer( self ):
		return self.theSingleObjectListBuffer


	def getSystemObjectListBuffer( self ):
		return self.theSystemObjectListBuffer


class LayoutBuffer:

	def __init__( self, aLayoutName ):
		self.theTextObjectListBuffer = ObjectListBuffer()
		self.theSystemObjectListBuffer = ObjectListBuffer()
		self.theConnectionObjectListBuffer = ObjectListBuffer()
		self.theName = aLayoutName
		self.thePropertyBuffer = PropertyListBuffer()

	def getName( self ):
		#returns name of layout
		return self.theName


	def getSystemObjectListBuffer( self ):
		return self.theSystemObjectListBuffer


	def getTextObjectListBuffer( self ):
		return self.theTextObjectListBuffer


	def getConnectionObjectListBuffer ( self ):
		return self.theConnectionObjectListBuffer


	def getPropertyList( self ):
		return self.thePropertyBuffer.getPropertyList()

	def getProperty( self, aPropertyName ):
		return self.thePropertyBuffer.getProperty( aPropertyName )

	def getPropertyBuffer ( self ):
		return self.thePropertyBuffer
