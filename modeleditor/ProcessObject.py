from EditorObject import *

class ProcessObject( EditorObject ):
	
	def __init__( self, aLayout, aFullID,  x,y, canvas= None ):
		pass


	def addConnectionLine( self, aName, aRingNumber, aVariableFullID, endX=None, endY=None ):
		#position by variablefullid takes precedence
		pass


	def deleteConnectionLine( self, aName ):
		pass


	def getConnectionLineList( self ):
		pass


	def show(self ):
		#render to canvas
		pass


	def reconnect( self, aConnectionName = None):
		pass


	def move( self, deltaX, deltaY):
		pass
