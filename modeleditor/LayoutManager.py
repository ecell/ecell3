

from ModelEditor import *
from Layout import *

class LayoutManager:

	def __init__( self, aModeleditor):
		self.theModelEditor = aModelEditor
		self.theLayoutMap = {}


	def createLayout( self ):
		pass


	def deleteLayout( self, aLayoutName ):
		pass


	def copyLayout( self, aLayoutName ):
		#returns layout buffer
		pass


	def saveLayouts( self, aFileName ):
		pass


	def loadLayouts( self, aFileName ):
		pass

	
	def pasteLayout( self, aLayoutBuffer ):
		pass


	def update( self, aType = None, anID = None ):
		pass


	def getUniqueLayoutName( self ):
		pass


