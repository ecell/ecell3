from LayoutBuffer import *
from BufferFactory import *

class LayoutBufferFactory:

	def __init__( self, aModelEditor, aLayoutManager):
		self.theModelEditor = aModelEditor
		self.theLayoutManager = aLayoutManager


	def createLayoutBuffer( self, aLayoutName ):
		aLayout = self.theLayoutManager.getLayout( aLayoutName )
		aLayoutBuffer = LayoutBuffer( aLayoutName )

		propertyBuffer = aLayoutBuffer.getPropertyBuffer()
		propertyList = aLayout.getPropertyList()
		for aProperty in propertyList:
			aValue = aLayout.getProperty( aProperty )
			propertyBuffer.createProperty( aProperty, aValue )


		# get all systemlist but copy only those whose parent is the layout self
		systemList = aLayout.getObjectList( OB_TYPE_SYSTEM )

		# add them to systemlist buffer
		for systemObjectID in systemList:
			systemObject = aLayout.getObject( systemObjectID )
			if systemObject.getParent() != aLayout:
				continue
			systemBuffer = self.createObjectBuffer( systemObjectID )
			aLayoutBuffer.getSystemObjectListBuffer().addObjectBuffer( systemBuffer )

		# copy all connections
		connectionList = aLayout.getObjectList( OB_TYPE_CONNECTION )
		for connectionID in connectionList:
			connectionBuffer = self.createObjectBuffer( connectionID )
			aLayoutBuffer.getConnectionObjectListBuffer().addObjectBuffer( connectionBuffer )
		# copy all textboxes whose parent is layout
		textList = aLayout.getObjectList( OB_TYPE_TEXT )
		for textID in connectionList:
			textObject = aLayout.getObject( textID )
			if textObject.getParent() != aLayout:
				continue
			textBuffer = self.createObjectBuffer( textID )
			aLayoutBuffer.getTextObjectListBuffer().addObjectBuffer( textBuffer )
		return aLayoutBuffer


	def createObjectBuffer( self, aLayoutName, anObjectID ):
		# DONOT COPY CONNECTIONS!!! Multiplexing will take care of that
		# but copy underlying entity
		aLayout = self.theLayoutManager.getLayout( aLayoutName )
		anObject = aLayout.getObject( anObjectID )
		aType = anObject.getProperty( OB_TYPE )
		if aType == OB_TYPE_SYSTEM:
			aBuffer = self.__createSystemObjectBuffer( anObject )
		else:
			aBuffer = self.__createObjectBuffer( anObject )
		return aBuffer
		
	
	def __createObjectBuffer ( self, anObject ):
		# TEXT, VARIABLE, PROCESS
		aType = anObject.getProperty( OB_TYPE )
		if aType == OB_TYPE_SYSTEM:
			aBuffer = aBuffer = SystemObjectBuffer( aSystemObject.getID() )
		else:
			aBuffer = ObjectBuffer( anObject.getID() )
		propertyBuffer = aBuffer.getPropertyBuffer()
		propertyList = anObject.getPropertyList()
		for aProperty in propertyList:
			aValue = anObject.getProperty( aProperty )
			# block connection copying
			if aProperty in [ VR_CONNECTIONLIST, PR_CONNECTIONLIST ]:
				continue
			propertyBuffer.createProperty( aProperty, aValue )
		if anObject.getProperty( OB_HASFULLID ):
			aFullID = anObject.getProperty( OB_FULLID )
			aBufferFactory = BufferFactory( self.theModelEditor.getModel() )
			if aType ==  OB_TYPE_SYSTEM:
				anEntityBuffer = aBufferFactory.createSystemListBuffer( [ aFullID ] )
		
			elif aType ==  OB_TYPE_VARIABLE:
				anEntityBuffer = aBufferFactory.createVariableListBuffer( [ aFullID ] )
		
			elif aType == OB_TYPE_PROCESS :
				anEntityBuffer = aBufferFactory.createProcessListBuffer( [ aFullID ] )

			aBuffer.setEntityBuffer( anEntityBuffer )
			
		return aBuffer


	def __createSystemObjectBuffer ( self, aSystemObject ):
		aBuffer = self.__createObjectBuffer( aSystemObject )
		childIDList = aSystemObject.getObjectList()
		aLayout = aSystemObject.getLayout()
		for aChildID in childIDList:
			childObject = aLayout.getObject( aChildID )
			aType = childObject.getProperty( OB_TYPE )
			if aType == OB_TYPE_SYSTEM:
				childSystemBuffer = self.__createSystemObjectBuffer( childObject )
				aBuffer.getSystemObjectListBuffer().addObjectBuffer( childSystemBuffer )
			elif aType != OB_TYPE_CONNECTION:
				childObjectBuffer = self.__createObjectBuffer( childObject )
				aBuffer.getSingleObjectListBuffer().addObjectBuffer( childObjectBuffer )
		return aBuffer


class LayoutBufferPaster:
	def __init__( self, aModelEditor, aLayoutManager):
		self.theModelEditor = aModelEditor
		self.theLayoutManager = aLayoutManager


	def pasteLayoutBuffer( self, aBuffer, newName = None ):
		# if newname is specified pastes with its name
		if newName == None:
			# get name from buffer
			newName = aBuffer.getName()
			if newName in self.theLayoutManager.getLayoutNameList():
				newName = "CopyOf" + newName
		# create layout
		self.theLayoutManager.createLayout( newName )
		
		aLayout = self.theLayoutManager.getLayout( newName )
		# paste properties
		aPropertyList = aBuffer.getPropertyList()
		for aProperty in aPropertyList:
			aValue = aBuffer.getProperty()
			aLayout.setProperty( aProperty, aValue )
		
		# paste systemlistbuffer
		self.__pasteObjectListBuffer( aBuffer.getSystemObjectListBuffer(), aLayout, newName )
		self.__pasteObjectListBuffer( aBuffer.getTextObjectListBuffer(), aLayout, newName )
		self.__pasteObjectListBuffer( aBuffer.getConnectionObjectListBuffer(), aLayout, newName )
		

	
	def __pasteObjectListBuffer( self, listBuffer, aParent, aLayoutName ):
		objectList = listBuffer.getObjectBufferList()
		for anObjectID in objectList:
			aBuffer = listBuffer.getObjectBuffer( anObjectID )
			self.pasteObjectBuffer( aLayoutName, aBuffer, None, None, aParent )
		
		
	
	def pasteObjectBuffer( self, aLayoutName, aBuffer, x = None, y = None, theParent  ):
		# dont create connections and entities
		aLayout = self.theLayoutManager.getLayout( aLayoutName )
		aType = aBuffer.getProperty( OB_TYPE )
		if aType == OB_TYPE_SYSTEM:
			self.__pasteSystemObjectBuffer( aLayout, aType, aBuffer, x, y, theParent )
		elif aType == OB_TYPE_CONNECTION:
			self.__pasteConnectionObjectBuffer( aLayout, aBuffer )
		else:
			self.__pasteObjectBuffer( aLayout, aType, aBuffer, x, y, theParent )



	def __pasteObjectBuffer( self, aLayout, aType, aBuffer, x, y, theParent ):
		if x == None:
			x = aBuffer.getProperty( OB_POS_X )
		if y == None:
			y = aBuffer.getProperty( OB_POS_Y )

		aFullID = None
		if aBuffer.getProperty( OB_HASFULLID ):
			aFullID = aBuffer.getProperty( OB_FULLID )
		# get ID
		anID = aBuffer.getID()
		if anID in aLayout.getObjectList( aType ):
			anID = aLayout.getUniqueObjectID()
		aLayout.createObject( anID, aType, aFullID, x, y, theParent  )
		anObject = aLayout.getObject( anID )
		propertyList = aBuffer.getPropertyList()
		for aProperty in propertyList:
			aValue = aBuffer.getProperty( aProperty )
			anObject.setProperty( aProperty, aValue )
		return anObject # for systems

		

	def __pasteSystemObjectBuffer( self,  aLayout, aType, aBuffer, x, y, theParent ):
		aSystem = self.__pasteObjectBuffer( aLayout, aType, aBuffer, x, y, theParent )
		aLayoutName = aLayout.getName( )
		
		self.__pasteObjectListBuffer( self, aBuffer.getSingleObjectListBuffer, aSystem, aLayoutName )
		self.__pasteObjectListBuffer( self, aBuffer.getSystemObjectListBuffer, aSystem, aLayoutName )
		
		
	def __pasteConnectionObjectBuffer( self, aLayout, aBuffer ):
		# makes sense only when copying layouts
		anID = aBuffer.getID()
		if anID in aLayout.getObjectList( OB_TYPE_CONNECTION ):
			anID = aLayout.getUniqueObjectID()
		processID = aBuffer.getProperty( CO_PROCESS_ATTACHED ).getID()
		variableID = aBuffer.getProperty( CO_VARIABLE_ATTACHED ).getID()
		processRing = aBuffer.getProperty( CO_PROCESS_RING )
		variableRing = aBuffer.getProperty( CO_VARIABLE_RING )
		aVarrefName = aBuffer.getProperty( CO_NAME )
		direction = aBuffer.getProperty( CO_DIRECTION )
		aLayout.createConnectionObject( anID, processID , variableID,  processRing, variableRing, direction, aVarrefName )
		anObject = aLayout.getObject( anID )
		propertyList = aBuffer.getPropertyList()
		for aProperty in propertyList:
			aValue = aBuffer.getProperty( aProperty )
			anObject.setProperty( aProperty, aValue )
