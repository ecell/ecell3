from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

class VariableObject( EditorObject ):
	
	def __init__( self, aLayout,objectID, aFullID,  x,y, canvas= None ):
		EditorObject.__init__( self, aLayout, objectID, x, y, canvas)
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_VARIABLE
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_VARIABLE
		#default dimensions
		self.thePropertyMap [ OB_LABEL ]=aFullID.split(':')[2]
		self.theLabel = self.thePropertyMap [ OB_LABEL ]
		self.thePropertyMap [ OB_MINLABEL ]=VAR_MINLABEL
		aVariableSD = VariableSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth
		self.thePropertyMap[ VR_CONNECTIONLIST ]= []
		reqWidth = aVariableSD.getRequiredWidth()
		reqHeight = aVariableSD.getRequiredHeight()
		

		self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
		if reqWidth<VAR_MINWIDTH:
			self.thePropertyMap [ OB_DIMENSION_X ]=VAR_MINWIDTH


		self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
		if reqHeight<VAR_MINHEIGHT:
			self.thePropertyMap [ OB_DIMENSION_Y]=VAR_MINHEIGHT
		aVariableSD.reCalculate()
		self.theSD = aVariableSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aVariableSD
		self.theVariableShapeList=['Rounded Rectangle']
		
	def reconnect( self ):
		pass

	def destroy( self ):
		connList = self.thePropertyMap[ VR_CONNECTIONLIST ][:]
		for aConnID in connList:
			self.theLayout.deleteObject( aConnID )
		EditorObject.destroy( self )


	def registerConnection( self, aConnectionID ):
		self.thePropertyMap[ VR_CONNECTIONLIST ].append( aConnectionID )

	def unRegisterConnection( self, aConnectionID ):
		
		self.thePropertyMap[ VR_CONNECTIONLIST].remove( aConnectionID )

	def estLabelWidth(self,newLabel):
		height,width=self.getGraphUtils().getTextDimensions(newLabel)
		return width+46

	def labelChanged( self,aPropertyValue ):
		#newLabel = aPropertyValue.split(':')[2]
		newLabel = aPropertyValue
		#totalWidth,limit=self.getLabelParam()
		#if totalWidth>limit:
		#	newLabel=self.truncateLabel(newLabel,totalWidth,limit)
		#	self.thePropertyMap[OB_LABEL]=newLabel
		self.theShape.labelChanged(self.getProperty(OB_LABEL))  
		
		

	def getAvailableVariableShape(self):
		return self.theVariableShapeList

	def parentMoved( self, deltax, deltay ):
		EditorObject.parentMoved( self, deltax, deltay )
		for aConID in self.thePropertyMap[VR_CONNECTIONLIST]:
			self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )


	def move( self, deltax, deltay ):
		EditorObject.move( self, deltax,deltay)
		for aConID in self.thePropertyMap[VR_CONNECTIONLIST]:
			self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

	def registerObject( self, anObject ):
		self.theObjectMap[anObject.getID()] = anObject


	def unregisterObject ( self, anObjectID ):
		self.theObjectMap.__delitem__( anObjectID )


	def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
		#first do a resize then a move
		# FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
		# resize must be sum of deltas
		self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
		self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown
		#print 'variable resize done' 
		self.theShape.resize(deltaleft + deltaright,deltaup + deltadown)
		if self.thePropertyMap[ VR_CONNECTIONLIST ] !=[]:
				
			for conn in self.getProperty(VR_CONNECTIONLIST):
				conobj =  self.theLayout.getObject(conn)
				(x, y) = self.getRingPosition(conobj.thePropertyMap[ CO_VARIABLE_RING ] )
				ringsize =  self.theSD.getRingSize()/2
				conobj.thePropertyMap[ CO_ENDPOINT2 ] = [ x + ringsize, y+ringsize]
				conobj.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].reCalculate()
				conobj.theShape.repaint()

		
	def getRingSize( self ):
		return self.theSD.getRingSize()


	def getRingPosition( self, ringCode ):
		#return absolute position of ring
		(xRing,yRing)=self.theSD.getShapeAbsolutePosition(ringCode)
		
		( x, y ) = self.getAbsolutePosition()
		
		return (x+xRing, y+yRing )
	
	
		

		

