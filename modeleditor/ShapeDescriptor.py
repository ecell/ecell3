from Constants import *

class ShapeDescriptor:

	def __init__( self ):
		self.theDescriptorList = []
		# DESCRIPTORLIST SHOULDNT BE DEPENDENT ON WIDTH AND HEIGHT BUT ONLY ON LABEL

	def getDescriptorList( self ):
		return self.theDescriptorList

	def getDescriptor ( self, shapeName ):
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == shapeName:
				return aDescriptor
		raise Exception (" Shape %s doesnot exist")


	def getRequiredWidth( self ):
		pass


	def getRequiredHeight( self ):
		pass


	def renameLabel (self, newLabel):
		# complexshape needs to be resized after this!!!
		pass


class SystemSD( ShapeDescriptor):

	def __init__( self, parentObject, graphUtils, aLabel ):
		ShapeDescriptor.__init__( self )
		self.parentObject = parentObject
		self.theGraphUtils = graphUtils
		self.theLabel = aLabel
		self.__createDescriptorList()


	def __createDescriptorList( self ):
		self.__calculateParams()
		self.theDescriptorList = [\
		#NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
		['frame', CV_RECT, SD_OUTLINE, SD_OUTLINE, 5, [ 0, 0, 1, 1, 0, 0, 0, 0 ],{} ],\
		['textarea',CV_RECT, SD_FILL, SD_FILL, 4, [ 0,0, 1, 0, self.olw, self.olw, -self.olw, self.olw + self.tx_height -1], {} ],\
		['labelline',CV_LINE,SD_FILL, SD_OUTLINE, 3, [  [ self.olw,0,self.tx_height+self.olw,0 , -self.olw, 1, self.tx_height+self.olw,0 ], self.olw ], {} ], \
		['drawarea', CV_RECT, SD_SYSTEM_CANVAS, SD_FILL, 4, [ 0,0, 1, 1, self.olw, self.tx_height+self.olw*2, -self.olw ,-self.olw ], {} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 3, [ self.theLabel, 0,0, self.olw +2, 2 ], {} ] ]


	def getInsideWidth( self ):
		self.__calculateParams()
		return self.insideWidth


	def getInsideHeight( self ):
		self.__calculateParams()
		return self.insideHeight


	def __calculateParams( self ):
		self.width = self.parentObject.thePropertyMap [ OB_DIMENSION_X ]
		self.height = self.parentObject.thePropertyMap [ OB_DIMENSION_Y ]
		
		(self.tx_height, self.tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel )
		self.tx_height += 2
		self.tx_width += 2
		#self.theLabel = self.theGraphUtils.truncateTextToSize( self.theLabel, self.width )
		self.olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )
		self.insideX = self.olw
		self.insideY = self.olw*2+self.tx_height
		self.insideWidth = self.width - self.insideX-self.olw
		self.insideHeight = self.height - self.insideY-self.olw


	def getRequiredWidth( self ):
		self.__calculateParams()
		return self.tx_width + self.olw*2 + 10


	def getRequiredHeight( self ):
		self.__calculateParams()
		return self.tx_height+ self.olw*3 + 10
	
