from Constants import *
from Utils import *
from Command import *

class LayoutCommand( Command ):


	def __checkArgs( self ):
		if type (self.theReceiver) == type(self):
			if self.theReceiver.__class__.__name__ == self.RECEIVER:
				return True

		return False


class CreateLayout(LayoutCommand):
	"""
	arg1: NAME
	"""
	RECEIVER = 'LayoutManager'
	ARGS_NO = 2
	NAME = 0
	SHOW = 1

	def checkArgs( self ):
		if not LayoutCommand.checkArgs(self):
			return False
		self.theName = self.theArgs[ self.NAME ]
		self.isShow = self.theArgs[ self.SHOW ]
		#check if layout name exists
		if self.theReceiver.doesLayoutExist(self.theName):
			return False
		return True

	
	def do( self ):
		self.theReceiver.createLayout( self.theName)
		if self.isShow:
			self.theReceiver.showLayout(self.theName)
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = [ DeleteLayout( self.theReceiver, self.theName ) ]


	def getAffected( self ):
		return (self.RECEIVER, None )



class DeleteLayout(LayoutCommand):
	"""
	arg1: NAME
	"""
	RECEIVER = 'LayoutManager'
	ARGS_NO = 1
	NAME = 0


	def checkArgs( self ):
		if not LayoutCommand.checkArgs(self):
			return False
		self.theName = self.theArgs[ self.NAME ]

		#check if layout name exists
		if not self.theReceiver.doesLayoutExist(self.theName):
			return False
		return True

	
	def do( self ):
		# prepare copy of layout
		layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.theName )
		layoutBuffer.setUndoFlag( True )
		# check if layout was shown and set show flag in pastelayout command accorddingly
		aLayout = self.theReceiver.getLayout( self.theName )
		
		self.theReverseCommandList = [ PasteLayout( self.theReceiver, layoutBuffer, None, aLayout.isShown() ) ]
		self.theReceiver.deleteLayout( self.theName)
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = None


	def getAffected( self ):
		return (self.RECEIVER, None )




class RenameLayout(LayoutCommand):
	"""
	arg1: NAME
	"""
	RECEIVER = 'LayoutManager'
	ARGS_NO = 2
	OLDNAME = 0
	NEWNAME = 1

	def checkArgs( self ):
		if not LayoutCommand.checkArgs(self):
			return False
		self.newName = self.theArgs[ self.NEWNAME ]
		self.oldName = self.theArgs[ self.OLDNAME ]
		#check if layout name exists
		if self.theReceiver.doesLayoutExist(self.newName):
			return False
		if not self.theReceiver.doesLayoutExist(self.oldName):
			return False
		return True


	def do( self ):
		self.theReceiver.renameLayout( self.oldName,self.newName)
		return True


	def createReverseCommand( self ):
		#self.theReverseCommandList = [ RenameLayout( self.theReceiver,  self.oldName, self.newName  ) ]
		self.theReverseCommandList = [ RenameLayout( self.theReceiver,  self.newName, self.oldName  ) ]


	def getAffected( self ):
		return (self.RECEIVER, None )


class CloneLayout(LayoutCommand):
	"""
	arg1: TEMPLATE
	"""
	RECEIVER = 'LayoutManager'
	ARGS_NO = 1
	TEMPLATE = 0

	def checkArgs( self ):
		if not LayoutCommand.checkArgs(self):
			return False
		self.theTemplate = self.theArgs[ self.TEMPLATE ]
		#check if layout name exists
		if not self.theReceiver.doesLayoutExist(self.theTemplate):
			return False
		return True


	def do(self):
		layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.theTemplate )
		newName = "copyOf" + self.theTemplate
		newName = self.theReceiver.getUniqueLayoutName( newName )
		self.theReceiver.theLayoutBufferPaster.pasteLayoutBuffer( layoutBuffer, newName )
		self.theReverseCommandList = [ DeleteLayout( self.theReceiver, newName ) ]
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = None


class PasteLayout(LayoutCommand):
	"""
	arg1: layoutbuffer
	arg2: new name if no new name, submit None
	"""
	RECEIVER = 'LayoutManager'
	ARGS_NO = 3
	BUFFER = 0
	NEWNAME = 1
	SHOW = 2

	def checkArgs( self ):
		if not LayoutCommand.checkArgs(self):
			return False
		self.theBuffer = self.theArgs[ self.BUFFER ]
		self.newName = self.theArgs[ self.NEWNAME ]
		self.isShow = self.theArgs[ self.SHOW ]
		return True


	def do(self):
		overWrite = False
		if self.newName == None:
			self.newName = self.theBuffer.getName()
		if self.theReceiver.doesLayoutExist(self.newName):
			#if self.theReceiver.theModelEditor.printMessage( "Do you want to overwrite layout %s"%self.newName ) = ME_RESULT_OK:
			# get copy of layout
			layoutBuffer = self.theReceiver.theLayoutBufferFactory.createLayoutBuffer( self.newName )
			layoutBuffer.setUndoFlag( True )
			#check if layougt was shown, and set flag in pastelayout command
			self.theReverseCommandList = [ PasteLayout( self.theReceiver, layoutBuffer, None, self.isShow ) ]
			self.theReceiver.deleteLayout( self.newName)
		else:
			self.theReverseCommandList = [ DeleteLayout( self.theReceiver, self.newName ) ]


		self.theReceiver.theLayoutBufferPaster.pasteLayoutBuffer( self.theBuffer, self.newName )
		if self.isShow:
			self.theReceiver.showLayout(self.newName)
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = None


	def getAffected( self ):
		return (self.RECEIVER, None )



class CreateObject(LayoutCommand):
	"""
	args: objectid, type, fullid, x, y
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 6
	OBJECTID = 0
	TYPE = 1
	FULLID = 2
	X = 3
	Y = 4
	PARENT = 5 

	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.theType = self.theArgs[ self.TYPE ]
		self.theFullID = self.theArgs[ self.FULLID ]
		self.x = self.theArgs[ self.X ]
		self.y = self.theArgs[ self.Y ]
		self.theParent = self.theArgs[ self.PARENT ]
		return True


	def do(self):
		self.theReceiver.createObject(self.objectID, self.theType, self.theFullID, self.x, self.y, self.theParent )
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = [ DeleteObject( self.theReceiver, self.objectID ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )


class DeleteObject(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 1
	OBJECTID = 0

	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		return True


	def do(self):
		objectBuffer = self.theReceiver.theLayoutBufferFactory.createObjectBuffer( self.theReceiver.getName(), self.objectID )

		self.theReverseCommandList = [ UndeleteObject( self.theReceiver, objectBuffer, None, None, None ) ]
		
		self.theReceiver.deleteObject(self.objectID)
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = None


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )


	

class SetObjectProperty(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 3
	OBJECTID = 0
	PROPERTYNAME = 1 # if None, get it from buffer
	NEWVALUE = 2 # if None get it from buffer


	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.propertyName = self.theArgs[ self.PROPERTYNAME ]
		self.newValue = self.theArgs[ self.NEWVALUE ]
		return True


	def do(self):
		# get object
		theObject = self.theReceiver.getObject( self.objectID )
		theObject.setProperty( self.propertyName, self.newValue )
		return True


	def createReverseCommand( self ):
		# store old value
		oldValue = copyValue( self.theReceiver.getObject(self.objectID).getProperty( self.propertyName ) )
		self.theReverseCommandList = [ SetObjectProperty( self.theReceiver, self.objectID, self.propertyName, oldValue ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



#class CutObject(LayoutCommand):
#	"""
#	args: objectid
#	"""
#	RECEIVER = 'Layout'
#	ARGS_NO = 1
#	OBJECTID = 0
#
#	def checkArgs( self ):
#		# no argument check - suppose call is right
#		self.objectID = self.theArgs[ self.OBJECTID ]
#		return True
#
#
#	def do(self):
#		objectBuffer = self.theReceiver.theLayoutBufferFactory.createObjectBuffer( self.theReceiver.getName(), self.objectID )
#		self.theReverseCommandList = [ PasteObject( self.theReceiver, objectBuffer, None, None ) ]
#		self.theReceiver.deleteObject(self.objectID)
#
#		return True
#
#
#	def createReverseCommand( self ):
#		self.theReverseCommandList = None
#
#
#	def getAffected( self ):
#		return (self.RECEIVER, self.theReceiver )



class PasteObject(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 4
	BUFFER = 0
	X = 1 # if None, get it from buffer
	Y = 2 # if None get it from buffer
	PARENT = 3 # cannot be None


	def checkArgs( self ):
		# no argument check - suppose call is right
		self.theBuffer = self.theArgs[ self.BUFFER ]
		self.x = self.theArgs[ self.X ]
		self.y = self.theArgs[ self.Y ]
		self.theParent = self.theArgs[ self.PARENT ]
		return True


	def do(self):
		self.theReceiver.theLayoutBufferPaster.pasteObjectBuffer( self.theReceiver.getName(), self.theBuffer, self.x, self.y, self.theParent )
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = [ DeleteObject( self.theReceiver, self.theBuffer.getID() ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )


class UndeleteObject(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 4
	BUFFER = 0
	X = 1 # if None, get it from buffer
	Y = 2 # if None get it from buffer
	PARENT = 3 # cannot be None


	def checkArgs( self ):
		# no argument check - suppose call is right
		self.theBuffer = self.theArgs[ self.BUFFER ]
		self.x = self.theArgs[ self.X ]
		self.y = self.theArgs[ self.Y ]
		self.theParent = self.theArgs[ self.PARENT ]
		self.theBuffer.setUndoFlag ( True )
		return True


	def do(self):
		self.theReceiver.theLayoutBufferPaster.pasteObjectBuffer( self.theReceiver.getName(), self.theBuffer, self.x, self.y, self.theParent )
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = [ ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



class MoveObject(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 4
	OBJECTID = 0
	NEWX = 1
	NEWY = 2
	NEWPARENT = 3

	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.newx = self.theArgs[ self.NEWX ]
		self.newy = self.theArgs[ self.NEWY ]
		#self.newParent = self.theArgs[ self.NEWPARENT ]
		self.newParent=None
		return True


	def do(self):
		self.theReceiver.moveObject( self.objectID, self.newx, self.newy, self.newParent )
		return True


	def createReverseCommand( self ):
		theObject = self.theReceiver.getObject( self.objectID )
		oldX = theObject.getProperty( OB_POS_X )
		oldY = theObject.getProperty( OB_POS_Y )
		self.theReverseCommandList = [ MoveObject( self.theReceiver, self.objectID, oldX, oldY ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



class ResizeObject(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 5
	OBJECTID = 0
	UP = 1 
	DOWN = 2
	LEFT = 3
	RIGHT = 4

	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.up = self.theArgs[ self.UP ]
		self.down = self.theArgs[ self.DOWN ]
		self.left = self.theArgs[ self.LEFT ]
		self.right = self.theArgs[ self.RIGHT ]
		return True


	def do(self):
		self.theReceiver.resizeObject( self.objectID, self.up, self.down, self.left, self.right )
		return True


	def createReverseCommand( self ):
		antiUp = -self.up
		antiDown = -self.down
		antiLeft = -self.left
		antiRight = -self.right
		self.theReverseCommandList = [ ResizeObject( self.theReceiver, self.objectID, antiUp, antiDown, antiLeft, antiRight ) ]
		


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



class CreateConnection(LayoutCommand):
	"""
	args: objectid
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 7
	OBJECTID = 0
	PROCESSOBJECTID = 1
	VARIABLEOBJECTID = 2 
	PROCESSRING = 3
	VARIABLERING = 4
	DIRECTION = 5
	VARREFNAME = 6

	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.processObjectID = self.theArgs[ self.PROCESSOBJECTID ]
		self.variableObjectID = self.theArgs[ self.VARIABLEOBJECTID ]
		self.processRing = self.theArgs[ self.PROCESSRING ]
		self.variableRing = self.theArgs[ self.VARIABLERING ]
		self.direction = self.theArgs[ self.DIRECTION ]
		self.varrefName = self.theArgs[ self.VARREFNAME ]
		return True


	def do(self):
		self.theReceiver.createConnectionObject( self.objectID, self.processObjectID, self.variableObjectID, self.processRing, self.variableRing, self.direction, self.varrefName )
		return True


	def createReverseCommand( self ):
		self.theReverseCommandList = [ DeleteObject( self.theReceiver, self.objectID ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



class RedirectConnection(LayoutCommand):
	"""
	args: anObjectID, newProcessObjectID, newVariableObjectID = None, newRing = None 
	"""
	RECEIVER = 'Layout'
	ARGS_NO = 6
	OBJECTID = 0
	NEWPROCESSOBJECTID = 1
	NEWVARIABLEOBJECTID = 2 
	NEWPROCESSRING = 3
	NEWVARIABLERING = 4
	NEWVARREFNAME = 5 # can be none
	
	def checkArgs( self ):
		# no argument check - suppose call is right
		self.objectID = self.theArgs[ self.OBJECTID ]
		self.newProcessObjectID = self.theArgs[ self.NEWPROCESSOBJECTID ]
		self.newVariableObjectID = self.theArgs[ self.NEWVARIABLEOBJECTID ]
		self.newProcessRing = self.theArgs[ self.NEWPROCESSRING ]
		self.newVariableRing = self.theArgs[ self.NEWVARIABLERING ]
		self.newVarrefName = self.theArgs[ self.NEWVARREFNAME ]
		return True


	def do(self):
		self.theReceiver.redirectConnectionObject( self.objectID, self.newProcessObjectID, self.newVariableObjectID, self.newProcessRing, self.newVariableRing )
		return True


	def createReverseCommand( self ):
		theObject = self.theReceiver.getObject( self.objectID )
		if self.newProcessObjectID == None:
			oldProcessObjectID = None
			oldProcessRing = None
		else:
			oldProcessObjectID = theObject.getProperty( CO_PROCESS_ATTACHED ).getProperty( OB_FULLID )
			oldProcessRing = theObject.getProperty( CO_PROCESS_RING )

		if self.newVariableObjectID == None:
			oldVariableObjectID = None
			oldVariableRing = None
		else:
			oldVariableObjectID = theObject.getProperty( CO_VARIABLE_ATTACHED )
			oldVariableRing = theObject.getProperty( CO_VARIABLE_RING )


		self.theReverseCommandList = [ RedirectConnectionObject( self.theReceiver, self.objectID, oldProcessObjectID, oldVariableObjectID, oldProcessRing, oldVariableRing ) ]


	def getAffected( self ):
		return (self.RECEIVER, self.theReceiver )



