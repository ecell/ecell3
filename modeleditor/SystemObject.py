
from EditorObject import *


class SystemObject(EditorObject):


	def __init__( self, aLayout, aFullID,  x, y , parentSystem ):
		pass


	def addItem( self, aObject, x=None, y=None ):
		pass


	def getID( self ):
		pass


	def resize( self , newWidth, newHeigth ):
		pass


	def getEmptySpace( self ):
		pass

	def show( self ):
		#render to canvas
		pass


	def getObjectList( self ):
		# return IDs
		pass
		
	def isWithinSystem( self, objectID ):
		#returns true if is within system
		pass
		
