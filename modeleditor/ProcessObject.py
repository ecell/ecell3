from EditorObject import *
from Constants import *
from ShapeDescriptor import *
from LayoutCommand import *
from Utils import *



class ProcessObject( EditorObject ):
	
	def __init__( self, aLayout, objectID, aFullID,  x,y, canvas= None ):
		EditorObject.__init__( self, aLayout, objectID,x, y, canvas )
		self.thePropertyMap[ OB_HASFULLID ] = True
		self.thePropertyMap [ OB_FULLID ] = aFullID

		self.thePropertyMap [ OB_SHAPE_TYPE ] = SHAPE_TYPE_PROCESS
		self.thePropertyMap [ OB_OUTLINE_WIDTH ] = 3
		self.thePropertyMap[ OB_TYPE ] = OB_TYPE_PROCESS
		self.thePropertyMap[ PR_CONNECTIONLIST ] = []
		#default dimensions
		self.theLabel = aFullID.split(':')[2]
		aProcessSD = ProcessSD(self, self.getGraphUtils(), self.theLabel )
		# first get text width and heigth

		reqWidth = aProcessSD.getRequiredWidth()
		reqHeight = aProcessSD.getRequiredHeight()
		
	
		self.thePropertyMap [ OB_DIMENSION_X ] = reqWidth
		if reqWidth<PRO_MINWIDTH:
			self.thePropertyMap [ OB_DIMENSION_X ]=PRO_MINWIDTH
		self.thePropertyMap [ OB_DIMENSION_Y ] = reqHeight

		if reqHeight<PRO_MINHEIGHT:
			self.thePropertyMap [ OB_DIMENSION_Y ]=PRO_MINHEIGHT


		if reqHeight<PRO_MINHEIGHT:
			self.thePropertyMap [ OB_DIMENSION_Y ]=PRO_MINHEIGHT

		self.connectionDragged = False
		
		self.theSD = aProcessSD
		self.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ] = aProcessSD

		self.theProcessShapeList=['Rectangle']


	def registerConnection( self, aConnectionID ):
		self.thePropertyMap[ PR_CONNECTIONLIST ].append( aConnectionID )

	def unRegisterConnection( self, aConnectionID ):
		self.thePropertyMap[PR_CONNECTIONLIST].remove( aConnectionID )


	def getAvailableProcessShape(self):
		return self.theProcessShapeList

	def destroy( self ):
		connList = self.thePropertyMap[ PR_CONNECTIONLIST ][:]
		for aConnID in connList:
			self.theLayout.deleteObject( aConnID )
		EditorObject.destroy( self )


	def parentMoved( self, deltax, deltay ):
		EditorObject.parentMoved( self, deltax, deltay )

		for aConID in self.thePropertyMap[PR_CONNECTIONLIST]:
			self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

	def move( self, deltax, deltay ):
		EditorObject.move( self, deltax,deltay)
		for aConID in self.thePropertyMap[PR_CONNECTIONLIST]:
			self.theLayout.getObject( aConID ).parentMoved( self.theID, deltax, deltay )

	def resize( self ,  deltaup, deltadown, deltaleft, deltaright  ):
		#first do a resize then a move
		# FIXME! IF ROOTSYSTEM RESIZES LAYOUT MUST BE RESIZED, TOOO!!!!
		# resize must be sum of deltas
		self.thePropertyMap[ OB_DIMENSION_X ] += deltaleft + deltaright
		self.thePropertyMap[ OB_DIMENSION_Y ] += deltaup + deltadown 	

	def buttonReleased( self ):
		EditorObject.buttonReleased( self )
		if self.connectionDragged:
			self.connectionDragged = False
			# get coordinates
			( endx, endy) = self.theGhostLine.getEndPoint( )
			self.theGhostLine.delete()
			self.theGhostLine = None
			# delete ghostline
			# CHECK IF VARIABLE IS CONNECTED
			( variableID, variableRing ) = self.theLayout.checkConnection( endx, endy, ME_VARIABLE_TYPE )
			if variableID == None:
				variableID = ( endx, endy )
				variableRing = None
			else:
				pass
				
			# create real line
			newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
			newVarrefName = self.__getNewVarrefID ()
			aCommand = CreateConnection( self.theLayout, newID, self.theID, variableID, self.theRingCode, variableRing, PROCESS_TO_VARIABLE, newVarrefName )
			self.theLayout.passCommand( [ aCommand ] )
			# newCon = self.theLayout.getObject( newID )
			# newCon.checkConnections()


	def ringDragged( self, aShapeName, deltax, deltay ):
		if self.connectionDragged:
			self.theGhostLine.moveEndPoint( deltax, deltay )
		else:
			self.theRingCode = aShapeName
			
			( x, y ) = self.getRingPosition( aShapeName )
			x += self.theSD.getRingSize()/2
			y += self.theSD.getRingSize()/2
			self.theGhostLine = GhostLine( self, aShapeName, x+deltax, y+deltay )
			self.connectionDragged = True
		
	def getRingSize( self ):
		return self.theSD.getRingSize()


	def getRingPosition( self, ringCode ):
		#return absolute position of ring
		(xRing,yRing)=self.theSD.getShapeAbsolutePosition(ringCode)
		
		( x, y ) = self.getAbsolutePosition()
		return (x+xRing, y+yRing )
	

	def labelChanged( self,aPropertyValue ):
		newLabel = aPropertyValue.split(':')[2]
		self.theShape.labelChanged(newLabel) 
		
		if self.thePropertyMap[ PR_CONNECTIONLIST ] !=[]:
				
			for conn in self.getProperty(PR_CONNECTIONLIST):
				conobj =  self.theLayout.getObject(conn)
				(x, y) = self.getRingPosition(conobj.thePropertyMap[ CO_PROCESS_RING ] )
				rsize = self.getRingSize()
				
				conobj.thePropertyMap[ CO_ENDPOINT1 ] = [ x +rsize/2, y+rsize/2 ]
				conobj.thePropertyMap[ OB_SHAPEDESCRIPTORLIST ].reCalculate()
				conobj.theShape.repaint()

	def __getNewVarrefID( self ):
		aFullID = self.getProperty( OB_FULLID )
		aFullPN = createFullPN ( aFullID, ME_PROCESS_VARREFLIST )
		aVarrefList = self.theLayout.theLayoutManager.theModelEditor.getModel().getEntityProperty( aFullPN )
		aVarrefNameList = []
		for aVarref in aVarrefList:
			aVarrefNameList.append( aVarref[ME_VARREF_NAME] )
		i= 0
		while 'X' +str(i) in aVarrefNameList:
			i += 1
		return 'X' + str( i )

	def getCursorType( self, aFunction, x, y, buttonPressed ):
		cursorType = EditorObject.getCursorType( self,  aFunction, x, y, buttonPressed  )
		if aFunction == SD_RING and buttonPressed:
			return CU_CONN_INIT
		return cursorType 


import gnome.canvas
class GhostLine:
	def __init__( self, parentObject, ringCode, endx, endy ):
		self.parentObject = parentObject
		self.x2 = endx
		self.y2 = endy
		(self.x1, self.y1) = self.parentObject.getRingPosition( ringCode )
		self.root = self.parentObject.theCanvas.getCanvas().root()
		self.theLine = self.root.add( gnome.canvas.CanvasLine, points=(self.x1, self.y1, self.x2, self.y2), last_arrowhead = gtk.TRUE,arrow_shape_a=5, arrow_shape_b=5, arrow_shape_c=3 )


	def moveEndPoint( self, deltax, deltay ):
		self.x2 += deltax
		self.y2 += deltay
		self.theLine.set_property( "points", (self.x1, self.y1, self.x2, self.y2) )


	def getEndPoint( self ):
		return ( self.x2, self.y2 )

	def registerObject( self, anObject ):
		self.theObjectMap[anObject.getID()] = anObject


	def unregisterObject ( self, anObjectID ):
		self.theObjectMap.__delitem__( anObjectID )


	def delete ( self ):
		self.theLine.destroy()
