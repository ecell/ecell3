


from LayoutManager import *
from ModelStore import *

class Layout:

	def __init__( self, aLayoutManager ):
		pass


	def setEditorWindow( self, anEditorWindow):
		pass


	def update( self, aType = None, anID = None ):
		pass


	#########################################
	#         	COMMANDS		#
	#########################################


	def createObject( self, anObjectType, x=None, y=None ):
		#returns objectID
		pass


	def deleteObject( self, anObjectID ):
		pass


	def getObjectList( self, anObjectType ):
		pass


	def setObjectProperty( self, aObjectID, aPropertyName, aPropertyValue ):
		pass


	def getObjectProperty( self, anObjectID, aPropertyName ):
		pass


	def getObjectPropertyList( self, anObjectID ):
		pass


	def copyObject( self, anObjectID ):
		pass


	def pasteObject( self, anObjectBuffer, x = None, y= None ):
		pass


	def moveObject(self, anObjectID, newX, newY ):
		pass



	def resizeObject( self, anObjectID, deltaTop, deltaBottom, deltaLeft, deltaRight ):
		# inward movement negative, outward positive
		pass


	def createConnectionObject( self, aProcessObjectID, aVariableObjectID=None,  startRing=None, endRing=None ):
		#returns ObjectID
		pass


	def redirectConnectionObject( self, anObjectID, aVariableObjectID = None, startRing = None, endRing = None ):
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


	def autoConnect( self, aProcessFullID, aVarrefName ):
		pass

	####################################################
	# 			OTHER			   #
	####################################################

	getUniqueObjectName( self, anObjectType ):
		pass

