
from EditorObject import *
from Constants import *

class SystemObject(EditorObject):


	def __init__( self, aLayout, objectID, aFullID,  x, y , parentSystem ):
		EditorObject.__init__( self, aLayout, objectID, x, y, parentSystem )
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID

		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_SYSTEM

		#default dimensions
		self.thePropertyMap [ OB_DIMENSION_X ] = 200
		self.thePropertyMap [ OB_DIMENSION_Y ] = 200
		if parentSystem == None:
			layoutDims = self.theLayout.getProperty( LO_SCROLL_REGION )
			self.thePropertyMap [ OB_DIMENSION_X ] = layoutDims[2] - layoutDims[0]
			self.thePropertyMap [ OB_DIMENSION_Y ] = layoutDims[3] - layoutDims[1]


		# first get text width and heigth
		self.theLabel = aFullID.split(':')[2]
		(tx_height, tx_width) = self.getGraphUtils().getTextDimensions( self.theLabel )
		self.theLabel = self.getGraphUtils().truncateTextToSize( self.theLabel, self.thePropertyMap [ OB_DIMENSION_X ] )
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X ] = self.thePropertyMap [ OB_DIMENSION_X ] - 4
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X ] = self.thePropertyMap [ OB_DIMENSION_X ] - 5 - tx_height
		olw = self.thePropertyMap[ OB_OUTLINE_WIDTH ]
		shapeDescriptorList = [\
		#NAME,TYPE,FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
		['frame', CV_RECT, SD_OUTLINE, SD_OUTLINE, 5, [ 0, 0, 1, 1, 0, 0, 0, 0],{} ],\
		['textarea',CV_RECT, SD_FILL, SD_FILL, 4, [ 0,0, 1, 0, olw, olw, -olw*2, tx_height], {} ],\
		['labelline',CV_LINE,SD_FILL, SD_OUTLINE, 3, [ [ [ olw,0,tx_height+olw*2,0 ],[ -olw*2, 1, tx_height+olw*2,0] ], 1 ], {} ], \
		['drawarea', CV_RECT, SD_FILL, SD_FILL, 4, [ 0,0, 1, 1, olw, tx_height+olw*2+1, -olw*2,-tx_height - olw ], {} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 3, [ self.theLabel,  ], {} ] ]
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = shapeDescriptorList


	def addItem( self, aObject, x=None, y=None ):
		pass


	def getID( self ):
		return self.theID


	def resize( self , newWidth, newHeigth ):
		pass


	def getEmptyPosition( self ):
		return ( 50,50 )


	def show( self ):
		#render to canvas
		EditorObject.show( self )



	def getObjectList( self ):
		# return IDs
		pass
		
	def isWithinSystem( self, objectID ):
		#returns true if is within system
		pass
		
	def getAbsolouteInsidePosition( self ):
		pass
