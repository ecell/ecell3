from Constants import *

class ShapeDescriptor:

	def __init__( self ):
		self.theDescriptorList = []

	def getDescriptorList( self ):
		return self.theDescriptorList

	def getDescriptor ( self, shapeName ):
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == shapeName:
				return aDescriptor
		raise Exception (" Shape %s doesnot exist")



class SystemSD( ShapeDescriptor):

	def __init__( self, parentObject, graphUtils, aLabel ):
		ShapeDescriptor.__init__( self )
		self.parentObject = parentObject
		self.theGraphUtils = graphUtils
		self.theLabel = aLabel
		
		self.width = self.parentObject.thePropertyMap [ OB_DIMENSION_X ]
		self.height = self.parentObject.thePropertyMap [ OB_DIMENSION_Y ]
		(tx_height, tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel )
		tx_height += 2
		tx_width += 2
		self.theLabel = self.theGraphUtils.truncateTextToSize( self.theLabel, self.width )
		olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )
		self.insideX = olw
		self.insideY = olw*2+tx_height
		self.insideWidth = self.width - self.insideX-olw
		self.insideHeight = self.height - self.insideY-olw

		self.theDescriptorList = [\
		#NAME, TYPE, FUNCTION, COLOR, Z, SPECIFIC, PROPERTIES  
		['frame', CV_RECT, SD_OUTLINE, SD_OUTLINE, 5, [ 0, 0, 1, 1, 0, 0, 0, 0 ],{} ],\
		['textarea',CV_RECT, SD_FILL, SD_FILL, 4, [ 0,0, 1, 0, olw, olw, -olw, olw + tx_height -1], {} ],\
		['labelline',CV_LINE,SD_FILL, SD_OUTLINE, 3, [  [ olw,0,tx_height+olw,0 , -olw, 1, tx_height+olw,0 ], olw ], {} ], \
		['drawarea', CV_RECT, SD_SYSTEM_CANVAS, SD_FILL, 4, [ 0,0, 1, 1, olw, tx_height+olw*2, -olw ,-olw ], {} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 3, [ self.theLabel, 0,0, olw +2, 2 ], {} ] ]

		print "frame", self.theDescriptorList[0][ SD_SPECIFIC ]
		print "textarea", self.theDescriptorList[1][ SD_SPECIFIC ]
		print "drawarea", self.theDescriptorList[3][ SD_SPECIFIC ]

	def getInsideWidth( self ):
		return self.insideWidth


	def getInsideHeight( self ):
		return self.insideHeight
