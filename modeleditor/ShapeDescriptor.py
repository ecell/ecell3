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
		self.theLabel = newLabel
		self.updateShapeDescriptor()
	

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



	def getLabel(self):
		return self.theLabel
	def updateShapeDescriptor(self):
		self.__createDescriptorList()
		
	def reCalculate( self ):
		self.__calculateParams()
class ProcessSD( ShapeDescriptor):

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
		['frame', CV_RECT, SD_FILL, SD_OUTLINE, 7, [ 0, 0, 0, 1, 0, 0, self.tx_width + 10 ,0],{} ],\
		['textarea',CV_RECT, SD_FILL, SD_FILL, 6, [ 0,0,0, 1, self.olw, self.olw, self.tx_width + 10 -self.olw, -self.olw  ], {} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 5, [ self.theLabel, 0,0, self.olw *2, self.olw*2 ],{} ],\
		[RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 5, [ 0, 0, 0, 0, (self.tx_width + 10)/2-self.olw, -self.olw,(self.tx_width + 10)/2+ self.olw, self.olw ],{} ],\
		[RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 5, [0, 0, 0, 0, (self.tx_width + 10)/2-self.olw, self.tx_height + 10-self.olw, (self.tx_width + 10)/2+self.olw, self.tx_height + 10+self.olw],{} ],\
		[RING_LEFT, CV_RECT, SD_RING, SD_OUTLINE, 5, [ 0, 0, 0, 0, -self.olw, (self.tx_height + 10)/2-self.olw, self.olw, (self.tx_height + 10)/2+self.olw ],{} ],\
		[RING_RIGHT, CV_RECT, SD_RING, SD_OUTLINE, 5, [0, 0,0, 0, (self.tx_width + 10)-self.olw, (self.tx_height + 10)/2 -self.olw, (self.tx_width + 10)+self.olw, (self.tx_height + 10)/2+self.olw ], {} ] ]


	
	def __calculateParams( self ):
		
		(self.tx_height, self.tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel )
		self.tx_height += 2
		self.tx_width += 2
		self.olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )


	def getRequiredWidth( self ):
		self.__calculateParams()
		return self.tx_width + self.olw*3


	def getRequiredHeight( self ):
		self.__calculateParams()
		return self.tx_height+ self.olw*3 

	def getLabel(self):
		return self.theLabel

	def updateDescriptorList(self):	
		self.__createDescriptorList()

	def getShapeAbsolutePosition(self,aShapeName):
		aDescList=self.getDescriptor(aShapeName)[5]
		x=aDescList[0]+aDescList[4] 
		y=aDescList[1]+aDescList[5] 
		return x,y

	def getRingSize( self ):
		return self.olw*2

	def updateShapeDescriptor(self):
		self.__createDescriptorList()
		
	def reCalculate( self ):
		self.__calculateParams()

class VariableSD( ShapeDescriptor):

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

		['leftframe',CV_ELL, SD_FILL, SD_OUTLINE, 6,  [ 0, 0, 0, 1, 0, 0, 40, 0 ],{} ],\
		['frame', CV_RECT, SD_FILL, SD_OUTLINE, 6, [ 0, 0, 0, 1, 20, 0, self.tx_width+20, 0 ],{} ],\
		['rightframe',CV_ELL, SD_FILL, SD_OUTLINE, 6,  [ 0, 0, 0, 1, self.tx_width, 0, self.tx_width+40, 0 ],{} ],\
		['leftcircle',CV_ELL, SD_FILL, SD_FILL, 5,  [ 0, 0, 0, 1, self.olw, self.olw,40-self.olw,0-self.olw ],{}],\
		['rect',CV_RECT, SD_FILL, SD_FILL, 5, [ 0, 0, 0, 1, 20-self.olw, self.olw, self.tx_width+20-self.olw, 0-self.olw], {} ],\
		['rightcircle',CV_ELL, SD_FILL, SD_FILL, 5, [ 0, 0, 0, 1, self.tx_width-self.olw, self.olw, self.tx_width+40-self.olw, 0-self.olw ],{} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ self.theLabel, 0,0, 20, self.olw*2], {} ],\
		[RING_TOP, CV_RECT, SD_RING, SD_OUTLINE, 3, [0, 0, 0, 0, (self.tx_width/2)+20-self.olw, -self.olw, (self.tx_width/2)+self.olw+20, self.olw ],{} ],\
		[RING_BOTTOM, CV_RECT, SD_RING, SD_OUTLINE, 3, [ 0,0, 0, 0, (self.tx_width/2)+20-self.olw, 30-self.olw, (self.tx_width/2)+self.olw+20, 30+self.olw],{} ],\
		[RING_LEFT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ 0, 0, 0, 0, -self.olw, 15-self.olw, self.olw, 15+self.olw],{} ],\
		[RING_RIGHT,CV_RECT, SD_RING, SD_OUTLINE, 3,  [ 0, 0, 0, 0, self.tx_width+40-self.olw, 15-self.olw, self.tx_width+40+self.olw, 15+self.olw],{} ]]
		


	def __calculateParams( self ):
		(self.tx_height, self.tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel )
		self.tx_height += 2
		self.tx_width += 2
		
		#self.theLabel = self.theGraphUtils.truncateTextToSize( self.theLabel, self.width )
		self.olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )
		

	def getRequiredWidth( self ):
		self.__calculateParams()
		return self.tx_width + self.olw*2 + 10 + 30


	def getRequiredHeight( self ):
		self.__calculateParams()
		return self.tx_height+ self.olw*3

	def getLabel(self):
		return self.theLabel

	def updateShapeDescriptor(self):
		self.__createDescriptorList()
		
	def reCalculate( self ):
		self.__calculateParams()

	def getShapeAbsolutePosition(self,aShapeName):
		aDescList=self.getDescriptor(aShapeName)[5]
		x=aDescList[0]+aDescList[4] 
		y=aDescList[1]+aDescList[5] 
		return x,y

	def getRingSize( self ):
		return self.olw*2

class TextSD( ShapeDescriptor ):
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
		['frame', CV_RECT, SD_FILL, SD_OUTLINE, 6, [ 0, 0, 1, 1, 0, 0,0, 0 ],{} ],\
		['rect',CV_RECT, SD_FILL, SD_FILL, 5, [ 0,0, 1, 1, self.olw, self.olw, -self.olw, -self.olw], {} ],\
		['text', CV_TEXT, SD_FILL, SD_TEXT, 4, [ self.theLabel, 0,0, self.olw*2, self.olw*2], {} ]]
		

	def __calculateParams( self ):
		(self.tx_height, self.tx_width) = self.theGraphUtils.getTextDimensions( self.theLabel)
		self.tx_height += 2
		self.tx_width += 2
		
		#self.theLabel = self.theGraphUtils.truncateTextToSize( self.theLabel, self.width )
		self.olw = self.parentObject.getProperty( OB_OUTLINE_WIDTH )


	def getRequiredWidth( self ):
		self.__calculateParams()
		return self.tx_width + self.olw*2 + 10 


	def getRequiredHeight( self ):
		self.__calculateParams()
		return self.tx_height+ self.olw*3 

	
	def getLabel(self):
		return self.theLabel

	def updateDescriptorList(self):	
		self.__createDescriptorList()

	
	def reCalculate( self ):
		self.__calculateParams()

class LineDescriptor:

	def __init__( self, graphUtils, parentObject ):
		self.theDescriptorList = []
		self.theGraphUtils = graphUtils
		self.parentObject = parentObject
		label = self.parentObject.getProperty( CO_NAME )
		firstArrow = self.parentObject.getProperty( CO_HASARROW1 )
		secondArrow = self.parentObject.getProperty( CO_HASARROW2 )
		self.up = 0
		linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
		self.theDescriptorList = [\
		[ "textbox", CV_TEXT, SD_FILL, SD_FILL, 0, [ label, 0,0,0,0 ], {}  ],\
		[ ARROWHEAD1, CV_LINE, SD_ARROWHEAD, SD_FILL, 1, [ 0,0,0,0, firstArrow, gtk.FALSE, linewidth ], {} ],\
		[ ARROWHEAD2, CV_LINE, SD_ARROWHEAD, SD_FILL, 1, [ 0,0,0,0 ,secondArrow, gtk.FALSE, linewidth], {} ] ]


	def renameLabel( self, newLabel ):
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == "textbox":
				aSpecific = aDescriptor[ SD_SPECIFIC ]
				aSpecific[0] = newLabel


	def getDescriptorList( self ):

		return self.theDescriptorList


	def getDescriptor ( self, shapeName ):
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == shapeName:
				return aDescriptor
		raise Exception (" Shape %s doesnot exist")


	def reCalculate( self ):
		self.calculateEndpoints()
		self.calculateSpecific()
		self.calculateTextPosition()


	def calculateTextPosition( self):
		pass

	def calculateEndpoints( self ):
		( self.x1, self.y1 ) = self.parentObject.getProperty( CO_ENDPOINT1 )
		
		dir1 = self.parentObject.getProperty( CO_DIRECTION1 )
		
		
		( self.x2, self.y2 ) = self.parentObject.getProperty( CO_ENDPOINT2 )
		
		dir2 = self.parentObject.getProperty( CO_DIRECTION2 )
		
		aProcessID = self.parentObject.getProperty( CO_PROCESS_ATTACHED ) 
		aProcessObj = self.parentObject.theLayout.getObject( aProcessID )
		self.aProWidth = aProcessObj.getProperty(OB_DIMENSION_X)
		self.aProHeight = aProcessObj.getProperty(OB_DIMENSION_Y)
		aVariableID = self.parentObject.getProperty( CO_VARIABLE_ATTACHED ) 

		ringsize = 0
		if aVariableID != None:
			aVariableObj = self.parentObject.theLayout.getObject( aVariableID )
			ringsize =  aVariableObj.theSD.getRingSize()/2
		
			
		# calculate 
		( self.insidex1, self.insidey1 ) = self.__calculateEndpoint( self.x1, self.y1, dir1 )
		( self.insidex2, self.insidey2 ) = self.__calculateEndpoint( self.x2, self.y2, dir2 )

		firstArrow = self.parentObject.getProperty( CO_HASARROW1 )
		secondArrow = self.parentObject.getProperty( CO_HASARROW2 )


		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == ARROWHEAD1:
				aSpecific = aDescriptor[ SD_SPECIFIC ]
				aSpecific [0] = self.x1
				aSpecific [1] = self.y1
				aSpecific [2] = self.insidex1
				aSpecific [3] = self.insidey1
				aSpecific [4] = firstArrow
			if aDescriptor[ SD_NAME ] == ARROWHEAD2:
				aSpecific = aDescriptor[ SD_SPECIFIC ]
				aSpecific [0] = self.x2+ringsize
				aSpecific [1] = self.y2+ringsize
				aSpecific [2] = self.insidex2
				aSpecific [3] = self.insidey2
				aSpecific [4] = secondArrow				
	def __calculateEndpoint( self, x, y, d ):
		
		if d & DIRECTION_UP == DIRECTION_UP:
			return ( x, y - ARROWHEAD_LENGTH )
		elif d & DIRECTION_DOWN == DIRECTION_DOWN:
			return ( x, y + ARROWHEAD_LENGTH )
		if d & DIRECTION_LEFT == DIRECTION_LEFT:
			return ( x - ARROWHEAD_LENGTH, y )
		elif d & DIRECTION_RIGHT == DIRECTION_RIGHT:
			return ( x + ARROWHEAD_LENGTH, y )

class corneredLineSD(LineDescriptor ):
	def __init__( self, parentObject, graphUtils ):

		LineDescriptor.__init__( self, graphUtils, parentObject )
		#NAME, TYPE, FUNCTION, COLOR, Z, POINTS, PROPERTIES 
		linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
		self.theDescriptorList.extend( [
		[ EXTEND1, CV_LINE, SD_FIXEDLINE, SD_FILL, 1, [ 0,0,0,0 ,gtk.FALSE, gtk.FALSE, linewidth], {} ],\
		[ EXTEND2, CV_LINE, SD_FIXEDLINE, SD_FILL, 1, [ 0,0,0,0 ,gtk.FALSE, gtk.FALSE, linewidth], {} ],\
		[ "straightlineL", CV_LINE, SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ],\
		[ "straightlineC", CV_LINE,  SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ],\
		[ "straightlineR", CV_LINE,  SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth],{} ] ] )
		
		self.reCalculate()


	#def reCalculate( self ):
	def calculateSpecific( self ):
		#self.calculateEndpoints()
	
		#get Q
		self.Q = self.__checkQ()
		#print self.Q
		
		if self.Q == 3 or self.Q == 4 :
			self.aProWidth=-self.aProWidth
		if self.Q == 4 or self.Q == 1:
			self.aProHeight = -self.aProHeight

		dir1 = self.parentObject.getProperty( CO_DIRECTION1 )
		dir2 = self.parentObject.getProperty( CO_DIRECTION2 )
		( self.extendx1, self.extendy1 ) = self.__calculateExtendpoint( self.insidex1, self.insidey1, dir1 )
		( self.extendx2, self.extendy2 ) = self.__calculateExtendpoint( self.insidex2, self.insidey2, dir2 )

		# calculate inside line:
		
		for aDescriptor in self.theDescriptorList:
				if aDescriptor[ SD_NAME ] == EXTEND1:
					aSpecific = aDescriptor[ SD_SPECIFIC ]
					aSpecific [0] = self.insidex1
					aSpecific [1] = self.insidey1
					aSpecific [2] = self.extendx1
					aSpecific [3] = self.extendy1
				if aDescriptor[ SD_NAME ] == EXTEND2:
					aSpecific = aDescriptor[ SD_SPECIFIC ]
					aSpecific [0] = self.insidex2
					aSpecific [1] = self.insidey2
					aSpecific [2] = self.extendx2
					aSpecific [3] = self.extendy2
				
				if self.Q == 1:
				
					if (dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT or dir1 & DIRECTION_DOWN == DIRECTION_DOWN) and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.a(aDescriptor)
					elif (dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT or dir1 & DIRECTION_UP == DIRECTION_UP) and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_DOWN == DIRECTION_DOWN and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.c(aDescriptor)

					elif dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_LEFT == DIRECTION_LEFT or dir2 & DIRECTION_UP == DIRECTION_UP ):
						self.b(aDescriptor)
			
					elif dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT ):
						self.d(aDescriptor)

					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.a(aDescriptor)
					
				if self.Q == 2:
					
					if dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT  and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.a(aDescriptor)
					if dir1 & DIRECTION_DOWN == DIRECTION_DOWN and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.a(aDescriptor)
					elif (dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT or dir1 & DIRECTION_DOWN == DIRECTION_DOWN) and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.b(aDescriptor)
					
					elif dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT== DIRECTION_RIGHT):
						self.d(aDescriptor)
					
					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT== DIRECTION_RIGHT):
						self.a(aDescriptor)
					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.c(aDescriptor)

				if self.Q == 3:
					if (dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT or dir1 & DIRECTION_DOWN == DIRECTION_DOWN or dir1 & DIRECTION_LEFT == DIRECTION_LEFT) and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.d(aDescriptor)
					
					elif (dir1 & DIRECTION_DOWN == DIRECTION_DOWN or dir1 & DIRECTION_LEFT == DIRECTION_LEFT or dir1 & DIRECTION_UP == DIRECTION_UP )and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.a(aDescriptor)
					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.c(aDescriptor)

				if self.Q ==4:
					if dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_LEFT == DIRECTION_LEFT and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.a(aDescriptor)
					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_UP == DIRECTION_UP and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.a(aDescriptor)
					elif dir1 & DIRECTION_DOWN == DIRECTION_DOWN and (dir2 & DIRECTION_UP == DIRECTION_UP or dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT):
						self.c(aDescriptor)
					elif dir1 & DIRECTION_DOWN == DIRECTION_DOWN and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.a(aDescriptor)
					elif dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT and (dir2 & DIRECTION_RIGHT == DIRECTION_RIGHT or dir2 & DIRECTION_UP == DIRECTION_UP):
						self.b(aDescriptor)
					elif dir1 & DIRECTION_RIGHT == DIRECTION_RIGHT and (dir2 & DIRECTION_DOWN == DIRECTION_DOWN or dir2 & DIRECTION_LEFT == DIRECTION_LEFT):
						self.d(aDescriptor)
					

	def a(self,aDescriptor):
		if aDescriptor[ SD_NAME ] == "straightlineL":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx2
			aSpecific [1] = self.extendy2
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy1

		if aDescriptor[ SD_NAME ] == "straightlineC":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1 
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy1
		if aDescriptor[ SD_NAME ] == "straightlineR":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1
			aSpecific [2] = self.extendx1
			aSpecific [3] = self.extendy1 
			
	def b(self,aDescriptor):
		if aDescriptor[ SD_NAME ] == "straightlineL":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy2
			aSpecific [2] = self.extendx1
			aSpecific [3] = self.extendy1
		if aDescriptor[ SD_NAME ] == "straightlineC":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy2 
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy2
		if aDescriptor[ SD_NAME ] == "straightlineR":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1
			aSpecific [2] = self.extendx1
			aSpecific [3] = self.extendy1
		
	def c(self,aDescriptor):

		if aDescriptor[ SD_NAME ] == "straightlineL":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1 -self.aProWidth
			aSpecific [1] = self.extendy1
			aSpecific [2] = self.extendx1-self.aProWidth
			aSpecific [3] = self.extendy2
			
		if aDescriptor[ SD_NAME ] == "straightlineC":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1- self.aProWidth
			aSpecific [1] = self.extendy2 
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy2
			
		if aDescriptor[ SD_NAME ] == "straightlineR":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1
			aSpecific [2] = self.extendx1 - self.aProWidth
			aSpecific [3] = self.extendy1
			
	def d(self,aDescriptor):
		if aDescriptor[ SD_NAME ] == "straightlineL":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1 
			aSpecific [2] = self.extendx1
			aSpecific [3] = self.extendy1 - self.aProHeight
			
		if aDescriptor[ SD_NAME ] == "straightlineC":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx1
			aSpecific [1] = self.extendy1 - self.aProHeight
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy1 - self.aProHeight
			
		if aDescriptor[ SD_NAME ] == "straightlineR":
			aSpecific = aDescriptor[ SD_SPECIFIC ]
			aSpecific [0] = self.extendx2
			aSpecific [1] = self.extendy1 - self.aProHeight
			aSpecific [2] = self.extendx2
			aSpecific [3] = self.extendy2
			

	
	def __calculateExtendpoint( self, x, y, d ):
		if d & DIRECTION_UP == DIRECTION_UP:
			return ( x, y )
		elif d & DIRECTION_DOWN == DIRECTION_DOWN:
			return ( x, y )
		if d & DIRECTION_LEFT == DIRECTION_LEFT:
			return ( x , y )
		elif d & DIRECTION_RIGHT == DIRECTION_RIGHT:
			return ( x , y )

	def __checkQ(self):
		
		if self.x2 >  self.x1 and self.y2 < self.y1:
			return 1
		elif self.x2 > self.x1 and self.y2 >self.y1:
			return 2
		elif self.x2 < self.x1 and self.y2 > self.y1:
			return 3
		elif self.x2 < self.x1 and self.y2< self.y1:
			return 4
		else:
			return 1

	def calculateTextPosition( self):
		distx=0
		disty=0
		x=0
		y=0
		absy1=0
		absy2=0
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == "textbox":
				aSpecific = aDescriptor[ SD_SPECIFIC ]
				label = self.parentObject.getProperty( CO_NAME )
				aSpecific[ TEXT_TEXT ] = label
				(labelwidth, labelheight) = self.theGraphUtils.getTextDimensions( label)
				if self.insidex1< self.insidex2:
					x1 = self.insidex1
					x2 = self.insidex2
					y1 = self.insidey1
					y2 = self.insidey2
				else:
					x1 = self.insidex2
					x2 = self.insidex1
					y1 = self.insidey2
					y2 = self.insidey1

				if y1 < y2:
					y -= labelheight
					x += labelheight
				else:
					y += 1

				
				distx=x2-x1
				if x1<x2 and y1==y1:
					x=x1+(distx/2)
					y=y1-5
				if x1==x2 and y1<y2:
					x=x1-5
					disty=y2-y1
					y=y1+(disty/2)
				if x1==x2 and y1>y2:
					x=x1-5
					disty=y1-y2
					y=y1+(disty/2)
				if x1<x2 and y2<y1:
					disty=y1-y2
					x=x1+distx/2
					absy1=abs(y1-x1)
					absy2=abs(y2-x1)
					if absy2<absy1:
						y=y2+5
					else:
						y=y1-5
					
				if x1<x2 and y2>y1:
					disty=y2-y1
					x=x1+distx/2
					absy1=abs(y1-x1)
					absy2=abs(y2-x1)
					if absy2<absy1:
						y=y2-5
					else:
						y=y1+5
				
				aSpecific [ TEXT_ABSX ] = x
				aSpecific [ TEXT_ABSY ] = y


class StraightLineSD( LineDescriptor ):

	def __init__( self, parentObject, graphUtils ):
		LineDescriptor.__init__( self, graphUtils, parentObject )
		#NAME, TYPE, FUNCTION, COLOR, Z, POINTS, PROPERTIES 
		linewidth = self.parentObject.getProperty( CO_LINEWIDTH )
		self.theDescriptorList.extend( [
		
		[ "straightline", CV_LINE, SD_MOVINGLINE, SD_FILL, 1, [ 0,0,0,0, gtk.FALSE, gtk.FALSE, linewidth], {} ]] )
		
		self.reCalculate()


	def calculateSpecific( self ):

		# calculate inside line:
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == "straightline":
				aSpecific = aDescriptor[ SD_SPECIFIC ]
				aSpecific [0] = self.insidex1
				aSpecific [1] = self.insidey1
				aSpecific [2] = self.insidex2
				aSpecific [3] = self.insidey2

	def calculateTextPosition( self):
		distx=0
		disty=0
		x=0
		y=0
		for aDescriptor in self.theDescriptorList:
			if aDescriptor[ SD_NAME ] == "textbox":
					aSpecific = aDescriptor[ SD_SPECIFIC ]
					label = self.parentObject.getProperty( CO_NAME )
					aSpecific[ TEXT_TEXT ] = label
					(labelwidth, labelheight) = self.theGraphUtils.getTextDimensions( label)
					if self.insidex1< self.insidex2:
						x1 = self.insidex1
						x2 = self.insidex2
						y1 = self.insidey1
						y2 = self.insidey2
					else:
						x1 = self.insidex2
						x2 = self.insidex1
						y1 = self.insidey2
						y2 = self.insidey1

					if y1 < y2:
						y -= labelheight
						x += labelheight
					else:
						y += 1

					distx=x2-x1
					if x1<x2 and y1==y1:
						x=x1+(distx/2)
						y=y1-5
					if x1==x2 and y1<y2:
						x=x1-5
						disty=y2-y1
						y=y1+(disty/2)
					if x1==x2 and y1>y2:
						x=x1-5
						disty=y1-y2
						y=y1+(disty/2)
					if x1<x2 and y2<y1:
						disty=y1-y2
						y=y2+(disty/2)
						x=x1+(distx/2)+5
					if x1<x2 and y2>y1:
						disty=y2-y1
						x=x1+(distx/2)+5
						y=y1+(disty/2)
				
					aSpecific [ TEXT_ABSX ] = x
					aSpecific [ TEXT_ABSY ] = y
	




		

