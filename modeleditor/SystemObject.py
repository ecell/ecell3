
from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *

class SystemObject(EditorObject):


	def __init__( self, aLayout, objectID, aFullID,  x, y , parentSystem ):
		EditorObject.__init__( self, aLayout, objectID, x, y, parentSystem )
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID
		self.theObjectMap = {}
		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_SYSTEM
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_SYSTEM
		#default dimensions
		self.thePropertyMap [ OB_DIMENSION_X ] = 200
		self.thePropertyMap [ OB_DIMENSION_Y ] = 200
		self.theLabel = aFullID

		self.theMaxShiftPos=0
		self.theMaxShiftNeg=0
		self.theorgdir=0
		self.theorgdelta=None
		
		self.thex1org=0
		self.they1org=0
		self.thex2org=0
		self.they2org=0
		self.theduporg=0
		self.theddownorg=0
		self.thedleftorg=0
		self.thedrightorg=0

		
		self.themaxshiftxneg=0
		self.themaxshiftyneg=0
		self.themaxshiftdiag=0

		aSystemSD = SystemSD(self, self.getGraphUtils(), self.theLabel )

		# first get text width and heigth
		reqWidth = aSystemSD.getRequiredWidth()
		reqHeight = aSystemSD.getRequiredHeight()
	

		if parentSystem.__class__.__name__ == 'Layout':
			layoutDims = self.theLayout.getProperty( LO_SCROLL_REGION )
			self.thePropertyMap [ OB_DIMENSION_X ] = layoutDims[2] - layoutDims[0]-1
			self.thePropertyMap [ OB_DIMENSION_Y ] = layoutDims[3]- layoutDims[1]-1
		else:
			if reqWidth > self.thePropertyMap [ OB_DIMENSION_X ]:
				self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
			if reqHeight > self.thePropertyMap [ OB_DIMENSION_Y ]:
				self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight
		
			if len(self.getObjectList())==0:	
				spaceleftX = self.parentSystem.getProperty( SY_INSIDE_DIMENSION_X ) - self.getProperty( OB_DIMENSION_X ) - self.getProperty( OB_POS_X )
				spaceleftY = self.parentSystem.getProperty( SY_INSIDE_DIMENSION_Y ) - self.getProperty( OB_DIMENSION_Y ) - self.getProperty( OB_POS_Y )
			

				spaceleft = min( spaceleftX, spaceleftY )/2
			
				if spaceleft > 200 and len(self.parentSystem.getObjectList())<2 :
					self.thePropertyMap [ OB_DIMENSION_Y ] += spaceleft 
					self.thePropertyMap [ OB_DIMENSION_X ] += spaceleft 


		self.theSD = aSystemSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aSystemSD
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X  ] = aSystemSD.getInsideWidth()
		self.thePropertyMap[ SY_INSIDE_DIMENSION_Y  ] = aSystemSD.getInsideHeight()

		self.theSystemShapeList=['Rectangle']
		
	def destroy( self ):
		for anObjectID in self.theObjectMap.keys()[:]:
			self.theLayout.deleteObject( anObjectID )
		EditorObject.destroy( self )

	def move( self, deltax, deltay ):
		EditorObject.move( self, deltax,deltay)
		for anObjectID in self.theObjectMap.keys():
			self.theObjectMap[ anObjectID ].parentMoved( deltax, deltay )

	def registerObject( self, anObject ):
		self.theObjectMap[anObject.getID()] = anObject


	def unregisterObject ( self, anObjectID ):
		self.theObjectMap.__delitem__( anObjectID )

	def parentMoved( self, deltax, deltay ):
		EditorObject.parentMoved( self, deltax, deltay )
		for anID in self.theObjectMap.keys():
			self.theLayout.getObject( anID ).parentMoved( deltax, deltay )


	def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
		#first do a resize then a move
		# FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
		# resize must be sum of deltas
		self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
		self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown 
		self.thePropertyMap[ SY_INSIDE_DIMENSION_X ] += deltaleft + deltaright
		self.thePropertyMap[ SY_INSIDE_DIMENSION_Y ] += deltaup + deltadown 
		self.theShape.resize( deltaleft + deltaright, deltaup + deltadown )
		self.move( -deltaleft, -deltaup )
		
	def setProperty(self, aPropertyName, aPropertyValue):
		if  self.theCanvas !=None:
			if aPropertyName == OB_DIMENSION_X :
				oldx = self.thePropertyMap[ OB_DIMENSION_X ]
				deltaright = aPropertyValue - oldx
				self.resize( 0,0,0,deltaright )
				return
			if aPropertyName == OB_DIMENSION_Y :
				oldy = self.thePropertyMap[ OB_DIMENSION_Y ]
				deltadown = aPropertyValue - oldy
				self.resize( 0,deltadown,0,0 )
				return
		EditorObject.setProperty(self, aPropertyName, aPropertyValue)
			

	def labelChanged( self,aPropertyValue ):
		self.theShape.labelChanged( aPropertyValue) 


	def getEmptyPosition( self ):
		return ( 50,50 )


	def show( self ):
		#render to canvas
		EditorObject.show( self )

	def addItem( self, absx,absy ):
		(offsetx, offsety ) = self.getAbsolutePosition()
		x = absx - (self.theSD.insideX + offsetx )
		y = absy - ( self.theSD.insideY + offsety )
		aSysPath = convertSysIDToSysPath( self.getProperty( OB_FULLID ) )
		aCommand = None
		buttonPressed = self.theLayout.getPaletteButton()
		if  buttonPressed == PE_SYSTEM:
			# create command
			aName = self.getModelEditor().getUniqueEntityName ( ME_SYSTEM_TYPE, aSysPath )
			aFullID = ':'.join( [ME_SYSTEM_TYPE, aSysPath, aName] )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_SYSTEM )

			# check boundaries
			maxAvailWidth,maxAvailHeight=self.getMaxDimensions(x,y)
			if ((maxAvailWidth>=SYS_MINWIDTH and maxAvailHeight>=SYS_MINHEIGHT) and self.isWithinParent(x,y,OB_TYPE_SYSTEM,None)):
				aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_SYSTEM, aFullID, x, y, self )
			else:
				# change cursor
				self.theShape.setCursor(CU_CROSS)


		elif buttonPressed == PE_PROCESS:
			# create command
			aName = self.getModelEditor().getUniqueEntityName ( ME_PROCESS_TYPE, aSysPath )
			aFullID = ':'.join( [ME_PROCESS_TYPE, aSysPath, aName] )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_PROCESS )
			
			# check boundaries
			if (not self.isOverlapSystem(x,y,OB_TYPE_PROCESS,None) and self.isWithinParent(x,y,OB_TYPE_PROCESS,None)):
				aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_PROCESS, aFullID, x, y, self )
			else:
				# change cursor
				self.theShape.setCursor(CU_CROSS)

		elif buttonPressed == PE_VARIABLE:
			# create command
			aName = self.getModelEditor().getUniqueEntityName ( ME_VARIABLE_TYPE, aSysPath )
			aFullID = ':'.join( [ME_VARIABLE_TYPE, aSysPath, aName] )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_VARIABLE)
			
			# check boundaries
			if (not self.isOverlapSystem(x,y,OB_TYPE_VARIABLE,None) and self.isWithinParent(x,y,OB_TYPE_VARIABLE,None)):
				aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_VARIABLE, aFullID, x, y, self )
			else:
				# change cursor
				self.theShape.setCursor(CU_CROSS)


		elif buttonPressed == PE_TEXT:
			pass
			'''
			#aName = self.getModelEditor().getUniqueEntityName (ME_SYSTEM_TYPE, aSysPath )
			objectID = self.theLayout.getUniqueObjectID( OB_TYPE_TEXT )

			# check boundaries
			if (not self.isOverlapSystem(x,y,OB_TYPE_TEXT,None) and self.isWithinParent(x,y,OB_TYPE_TEXT,None)):
				aCommand = CreateObject( self.theLayout, objectID, OB_TYPE_TEXT,None,x, y, self)
			else:
				# change cursor
				self.theShape.setCursor(CU_CROSS)
			'''
		elif buttonPressed == PE_SELECTOR:
			self.doSelect()
		elif buttonPressed == PE_CUSTOM:
			pass

		if aCommand != None:
			self.theLayout.passCommand( [aCommand] )

	def getObjectList( self ):
		# return IDs
		return self.theObjectMap.keys()


	def getObject( self, anObjectID ):
		return self.theObjectMap[ anObjectID ]
		
	def isWithinSystem( self, objectID ):
		#returns true if is within system
		pass
		
	def getAbsoluteInsidePosition( self ):
		( x, y ) = self.getAbsolutePosition()
		return ( x+ self.theSD.insideX, y+self.theSD.insideY )
	
	#def isResizeOk(self,x,y)):

	def getCursorType( self, aFunction, x, y, buttonPressed ):
		try:
			aCursorType = EditorObject.getCursorType( self, aFunction, x, y, buttonPressed )
			if aFunction == SD_SYSTEM_CANVAS and self.theLayout.getPaletteButton() != PE_SELECTOR:
				aCursorType = CU_ADD
			elif aFunction == SD_OUTLINE:
				olw=self.getProperty( OB_OUTLINE_WIDTH )
				direction = self.getDirection( x, y )
				xmaxdiagpos= 0 
				xmaxdiagneg=0
				ymaxdiagneg=0 
				maxpos=0
				maxneg=0
				if direction == DIRECTION_UP | DIRECTION_LEFT:
					xmaxdiagpos=self.getMaxDiagShiftPos(direction)
					xmaxdiagneg,ymaxdiagneg=self.getMaxDiagShiftNeg(direction)
					if xmaxdiagpos>olw or xmaxdiagneg<-olw or ymaxdiagneg<-olw:
						aCursorType = CU_RESIZE_TOP_LEFT
					else:
						aCursorType = CU_CROSS

				elif direction == DIRECTION_UP:
					maxpos= self.getMaxShiftPos(self,direction)
					maxneg= -self.getMaxShiftNeg(direction)
					if maxpos>olw or maxneg<-olw:
						aCursorType = CU_RESIZE_TOP
					else:
						aCursorType = CU_CROSS

				elif direction == DIRECTION_UP | DIRECTION_RIGHT:
					xmaxdiagpos=self.getMaxDiagShiftPos(direction)
					xmaxdiagneg,ymaxdiagneg=self.getMaxDiagShiftNeg(direction)
					if xmaxdiagpos>olw or xmaxdiagneg<-olw or ymaxdiagneg<-olw:
						aCursorType = CU_RESIZE_TOP_RIGHT
					else:
						aCursorType = CU_CROSS
				

				elif direction == DIRECTION_RIGHT:
					maxpos= self.getMaxShiftPos(self,direction)
					maxneg= -self.getMaxShiftNeg(direction)
					if maxpos>olw or maxneg<-olw:
						aCursorType = CU_RESIZE_RIGHT
					else:
						aCursorType = CU_CROSS

				elif direction == DIRECTION_LEFT:
					maxpos= self.getMaxShiftPos(self,direction)
					maxneg= -self.getMaxShiftNeg(direction)
					if maxpos>olw or maxneg<-olw:
						aCursorType = CU_RESIZE_LEFT
					else:
						aCursorType = CU_CROSS
				

				elif direction == DIRECTION_DOWN:
					maxpos= self.getMaxShiftPos(self,direction)
					maxneg= -self.getMaxShiftNeg(direction)
					if maxpos>olw or maxneg<-olw:
						aCursorType = CU_RESIZE_BOTTOM
					else:
						aCursorType = CU_CROSS
				

				elif direction == DIRECTION_DOWN | DIRECTION_RIGHT:
					xmaxdiagpos=self.getMaxDiagShiftPos(direction)
					xmaxdiagneg,ymaxdiagneg=self.getMaxDiagShiftNeg(direction)
					if xmaxdiagpos>olw or xmaxdiagneg<-olw or ymaxdiagneg<-olw:
						aCursorType = CU_RESIZE_BOTTOM_RIGHT
					else:
						aCursorType = CU_CROSS
				

				elif direction == DIRECTION_DOWN | DIRECTION_LEFT:
					xmaxdiagpos=self.getMaxDiagShiftPos(direction)
					xmaxdiagneg,ymaxdiagneg=self.getMaxDiagShiftNeg(direction)
					if xmaxdiagpos>olw or xmaxdiagneg<-olw or ymaxdiagneg<-olw:
						aCursorType = CU_RESIZE_BOTTOM_LEFT
					else:
						aCursorType = CU_CROSS
		except:
			pass
		return aCursorType


	def getDirection( self, absx, absy ):
		olw = self.getProperty( OB_OUTLINE_WIDTH )
		width = self.getProperty( OB_DIMENSION_X )
		height = self.getProperty( OB_DIMENSION_Y )
		(offsetx, offsety ) = self.getAbsolutePosition()
		x = absx- offsetx
		y = absy - offsety

		direction = 0
		#leftwise direction:
		if x <= olw:
			direction |= DIRECTION_LEFT
		
		# rightwise direction
		elif x>= width -olw:
			direction |= DIRECTION_RIGHT

		# upwards direction
		if y <= olw:
			direction |= DIRECTION_UP
			

		# downwards direction
		elif y>= height - olw:
			direction |= DIRECTION_DOWN
			
			
		return direction

##################################################################################################
	def diagonalDragged(self,deltax, deltay, absx, absy,direction):
		dx=deltax
		dy=deltay
		deltaup = 0
		deltadown = 0
		deltaleft = 0
		deltaright = 0
		
		if direction==0 and not self.theShape.getFirstDrag():
			return

		if self.theShape.getFirstDrag() and not self.theShape.getDragBefore() :
			self.thex1org=self.thePropertyMap[OB_POS_X]
			self.they1org=self.thePropertyMap[OB_POS_Y]
			self.thex2org=self.thex1org+self.thePropertyMap[OB_DIMENSION_X]
			self.they2org=self.they1org+self.thePropertyMap[OB_DIMENSION_Y]
			self.themaxshiftdiag=self.getMaxDiagShiftPos(direction)
			self.themaxshiftxneg,self.themaxshiftyneg=self.getMaxDiagShiftNeg(direction)
			self.theorgdir=direction
			self.theShape.setDragBefore(True)
			self.theShape.setFirstDrag(False)
			if dx>0 or dy>0:
				self.theorgdelta='pos'
			elif dx<0 or dy<0:
				self.theorgdelta='neg'

		

		if direction==0 and self.theorgdir!=0:	
			if self.theShape.getIsButtonPressed() :
				direction=self.theorgdir
		
		if self.theShape.getIsButtonPressed() and self.theorgdir==direction:
			if (dx<0 or dy<0) and self.theorgdelta=='pos':
				self.themaxshiftxneg,self.themaxshiftyneg=self.getMaxDiagShiftNeg(direction)
				self.theorgdelta='neg'
			elif (dx>0 or dy>0) and self.theorgdelta=='neg':	
				self.themaxshiftdiag=self.getMaxDiagShiftPos(direction)
				self.theorgdelta='pos'
		
		if direction==DIRECTION_BOTTOM_RIGHT :
			if dx>0  or dy>0:
				if self.themaxshiftdiag>0:
					if dx>0:
						if self.themaxshiftdiag>dx:
							deltaright=dx
							deltadown=dx
							self.themaxshiftdiag-=dx
					elif dy>0:
						if self.themaxshiftdiag> dy  :
							deltaright=dy
							deltadown=dy
							self.themaxshiftdiag-=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir: 
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return	
				
			elif dx<0  or dy<0:
				if self.themaxshiftxneg<0 and self.themaxshiftyneg<0 :
					if dx<0:
						if self.themaxshiftxneg< dx and self.themaxshiftyneg<dy :
							deltaright=dx
							deltadown=dx
							self.themaxshiftxneg-=dx
							self.themaxshiftyneg-=dx
					elif dy<0:
						if self.themaxshiftxneg< dy and self.themaxshiftyneg<dy :
							deltaright=dy
							deltadown=dy
							self.themaxshiftxneg-=dy
							self.themaxshiftyneg-=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir: 	
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return	
				
		elif direction==DIRECTION_BOTTOM_LEFT :	
			if dx>0 or dy<0:
				if self.themaxshiftxneg<0 and self.themaxshiftyneg<0 :
					if dx>0:
						if self.themaxshiftxneg<-dx and self.themaxshiftyneg<-dx :
							deltaleft=-dx
							deltadown=-dx
							self.themaxshiftxneg+=dx
							self.themaxshiftyneg+=dx
					elif  dy<0:
						if self.themaxshiftxneg<dy and self.themaxshiftyneg<dy :
							deltaleft=dy
							deltadown=dy
							self.themaxshiftxneg-=dy
							self.themaxshiftyneg-=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return
			elif dx<0 or dy>0:
				if self.themaxshiftdiag>0 :
					if dx<0:
						if self.themaxshiftdiag>-dx  :
							deltaleft=-dx
							deltadown=-dx
							self.themaxshiftdiag+=dx
					elif dy>0:
						if self.themaxshiftdiag>dy :
							deltaleft=dy
							deltadown=dy
							self.themaxshiftdiag-=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return					
			
		elif direction==DIRECTION_TOP_RIGHT :
			if dx>0 or dy<0:
				if self.themaxshiftdiag>0 :
					if dx>0:
						if self.themaxshiftdiag>dx :
							deltaup=dx
							deltaright=dx
							self.themaxshiftdiag-=dx
					elif dy<0:
						if self.themaxshiftdiag>-dy :
							deltaright=-dy
							deltaup=-dy
							self.themaxshiftdiag+=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return	
			elif dx<0 or dy>0:
				if self.themaxshiftxneg<0 and self.themaxshiftyneg<0 :
					if dx<0:
						if self.themaxshiftxneg<dx and self.themaxshiftyneg<dx :
							deltaright=dx
							deltaup=dx
							self.themaxshiftxneg-=dx
							self.themaxshiftyneg-=dx
					elif  dy>0:
						if self.themaxshiftxneg<-dy and self.themaxshiftyneg<-dy :
							deltaright=-dy
							deltaup=-dy
							self.themaxshiftxneg+=dy
							self.themaxshiftyneg+=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return
						
		elif direction==DIRECTION_TOP_LEFT:
			if dx<0 or dy<0: 
				if self.themaxshiftdiag>0 :
					if dx<0:
						if self.themaxshiftdiag>-dx :
							deltaup=-dx
							deltaleft=-dx
							self.themaxshiftdiag+=dx
					elif dy<0:
						if self.themaxshiftdiag>-dy :
							deltaup=-dy
							deltaleft=-dy
							self.themaxshiftdiag+=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return		
			elif dx>0 or dy>0:
				if self.themaxshiftxneg<0 and self.themaxshiftyneg<0 :
					if dx>0:
						if self.themaxshiftxneg<-dx and self.themaxshiftyneg<-dx :
							deltaleft=-dx
							deltaup=-dx
							self.themaxshiftxneg+=dx
							self.themaxshiftyneg+=dx
					elif  dy>0:
						if self.themaxshiftxneg<-dy and self.themaxshiftyneg<-dy :
							deltaleft=-dy
							deltaup=-dy
							self.themaxshiftxneg+=dy
							self.themaxshiftyneg+=dy
					if self.theShape.getIsButtonPressed():
						if direction==self.theorgdir:
							self.resize( deltaup, deltadown, deltaleft, deltaright)
							self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return

	def outlineDragged( self, deltax, deltay, absx, absy):
		deltaup = 0
		deltadown = 0
		deltaleft = 0
		deltaright = 0
		delta=0

		if not self.theShape.getIsButtonPressed() :
			self.thedleftorg=self.thePropertyMap[OB_POS_X]-self.thex1org
			self.thedrightorg=self.thePropertyMap[OB_POS_X]+self.thePropertyMap[OB_DIMENSION_X]-self.thex2org
			self.theduporg=self.thePropertyMap[OB_POS_Y]-self.they1org
			self.theddownorg=self.thePropertyMap[OB_POS_Y]+self.thePropertyMap[OB_DIMENSION_Y]-self.they2org
			self.resize( self.theduporg, -self.theddownorg, self.thedleftorg, -self.thedrightorg)
		
			aCommand = ResizeObject( self.theLayout, self.theID, -self.theduporg, self.theddownorg, -self.thedleftorg, self.thedrightorg )
			self.theLayout.passCommand( [aCommand] )
		
		if self.theShape.getOutlineDragged():
			direction = self.getDirection( absx, absy )
		
		if direction & DIRECTION_LEFT == DIRECTION_LEFT:
			deltaleft = -deltax
			delta=deltaleft
		elif direction & DIRECTION_RIGHT == DIRECTION_RIGHT:
			deltaright = deltax
			delta=deltaright
		if direction & DIRECTION_UP == DIRECTION_UP:
			deltaup = - deltay
			delta=deltaup
		elif direction & DIRECTION_DOWN == DIRECTION_DOWN:
			deltadown=deltay
			delta=deltadown
		if direction & DIRECTION_BOTTOM_RIGHT==DIRECTION_BOTTOM_RIGHT or direction & DIRECTION_BOTTOM_LEFT == DIRECTION_BOTTOM_LEFT:
			self.diagonalDragged(deltax, deltay, absx, absy,direction)
			return
		elif direction==DIRECTION_TOP_RIGHT or direction==DIRECTION_TOP_LEFT:	
			self.diagonalDragged(deltax, deltay, absx, absy,direction)
			return
	
		#if direction==0 and not self.theShape.getFirstDrag():
		#	return
		
		
	
		#FIXMEparentSystem boundaries should be watched!!!
		if self.theShape.getFirstDrag() and not self.theShape.getDragBefore() :
			self.thex1org=self.thePropertyMap[OB_POS_X]
			self.they1org=self.thePropertyMap[OB_POS_Y]
			self.thex2org=self.thex1org+self.thePropertyMap[OB_DIMENSION_X]
			self.they2org=self.they1org+self.thePropertyMap[OB_DIMENSION_Y]
			self.theMaxShiftPos=self.getMaxShiftPos(self,direction)
			self.theMaxShiftNeg=-(self.getMaxShiftNeg(direction))
			self.theorgdir=direction
			self.theShape.setFirstDrag(False)
			self.theShape.setDragBefore(True)
			if delta>0:
				self.theorgdelta='pos'
			else:
				self.theorgdelta='neg'


		if direction==0 and self.theorgdir!=0:	
			if self.theShape.getIsButtonPressed() :
				direction=self.theorgdir
	
		if self.theShape.getIsButtonPressed() and self.theorgdir==direction:
			if delta<0 and self.theorgdelta=='pos':
				self.theMaxShiftNeg=-(self.getMaxShiftNeg(direction))
				self.theorgdelta='neg'
			elif delta>0 and self.theorgdelta=='neg':	
				self.theMaxShiftPos=self.getMaxShiftPos(self,direction)
				self.theorgdelta='pos'	
		
		if delta>0 and self.theMaxShiftPos>0 :
			if self.theMaxShiftPos> delta :
				if self.theShape.getIsButtonPressed():
					if direction==self.theorgdir:
						self.theMaxShiftPos-=delta
						self.resize( deltaup, deltadown, deltaleft, deltaright)
						self.adjustCanvas(deltax,deltay)
					else:
						return		
				else:
					return				
			else:
				return

		elif delta<0 and self.theMaxShiftNeg<0 :
			if self.theMaxShiftNeg<delta :
				if self.theShape.getIsButtonPressed():
					if direction==self.theorgdir:
						self.theMaxShiftNeg-=delta
						self.resize( deltaup, deltadown, deltaleft, deltaright)
						self.adjustCanvas(deltax,deltay)
					else:
						return
				else:
					return

	

	def getAvailableSystemShape(self):
			return self.theSystemShapeList

