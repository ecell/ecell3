from Constants import *
from LayoutCommand import *

class PackingStrategy:

	def __init__( self, aLayout ):
		self.theLayout = aLayout


	def autoMoveObject( self, systemFullID, objectID ):
		# return cmdlist: delete or move + resize command
		cmdList = []
		# get Object of systemFullID
		systemList = self.theLayout.getObjectList( OB_TYPE_SYSTEM )
		systemObject = None
		for aSystemObject in systemList:
			if aSystemObject.getProperty( OB_FULLID ) == systemFullID:
				systemObject = aSystemObject
				break
		if systemObject == None:
			# create delete command
			delCmd = DeleteObject ( self.theLayout, objectID )
			cmdList.append( delCmd )
		else:
			anObject = self.theLayout.getObject( objectID )
			# get dimensions of object
			objectWidth = anObject.getProperty( OB_DIMENSION_X )
			objectHeigth = anObject.getProperty( OB_DIMENSION_Y )
			
			# get inside dimensions of system
			systemWidth = systemObject.getProperty ( SY_INSIDE_DIMENSION_X )
			systemHeigth = systemObject.getProperty ( SY_INSIDE_DIMENSION_Y )

			# resize if necessary
			resizeNeeded = False
			oldObjectWidth = objectWidth
			oldObjectHeigth = objectHeigth
			if objectWidth >= systemWidth:
				resizeNeeded = True
				objectWidth = systemWidth /2
				if objectWidth < OB_MIN_WIDTH:
					objectWidth = OB_MIN_WIDTH
		
			
			if objectHeigth >= systemHeigth:
				resizeNeeded = True
				objectHeigth = systemHeigth /2
				if objectHeigth < OB_MIN_HEIGTH:
					objectHeigth = OB_MIN_HEIGTH

			if resizeNeeded:
				cmdList.append( ResizeObject( self.theLayout, objectID, 0, objectHeigth - oldObjectHeigth, 0, objectWidth - oldObjectWidth ) )
			# get rnd coordinates
			leewayX = systemWidth - objectWidth
			leewayY = systemHeigth - objectHeigth
			import random
			rGen = random.Random(leewayX)
			posX = rGen.uniform(0,leewayX)
			posY = rGen.uniform(0,leewayY)

			# create cmd
			cmdList.append( MoveObject( self.theLayout, objectID, posX, posY, systemObject ) )

		return cmdList



	def autoConnect( self, processObjectID, variableObjectID,varrefName ):
		# return cmdlist
		cmdList = []
		
		aProObject = self.theLayout.getObject( processObjectID )
		aVarObject = self.theLayout.getObject( variableObjectID )

		# get dimensions of object and x, y pos
		aProObjectWidth = aProObject.getProperty( OB_DIMENSION_X )
		aProObjectHeigth = aProObject.getProperty( OB_DIMENSION_Y )
		(aProObjectX1,aProObjectY1)=aProObject.getAbsolutePosition()
		aProObjectX2 = aProObjectX1 + aProObjectWidth
		aProObjectY2 = aProObjectY1 + aProObjectHeigth
		aProObjectXCenter = aProObjectX1 + aProObjectWidth/2
		aProObjectYCenter = aProObjectY1 + aProObjectHeigth/2

		aVarObjectWidth = aVarObject.getProperty( OB_DIMENSION_X )
		aVarObjectHeigth = aVarObject.getProperty( OB_DIMENSION_Y )
		(aVarObjectX1,aVarObjectY1)=aVarObject.getAbsolutePosition()
		aVarObjectXCenter = aVarObjectX1 +aVarObjectWidth/2
		aVarObjectYCenter = aVarObjectY1 +aVarObjectHeigth/2

		if aVarObjectXCenter >= aProObjectXCenter and aVarObjectYCenter <= aProObjectYCenter:
			if aVarObjectYCenter >=aProObjectY1:
				processRing =RING_RIGHT
				variableRing =RING_LEFT
			if aVarObjectYCenter <= aProObjectY1:
				processRing =RING_TOP
				variableRing =RING_BOTTOM
		if aVarObjectXCenter >= aProObjectXCenter and aVarObjectYCenter >= aProObjectYCenter:
			if aVarObjectYCenter <= aProObjectY2:
				processRing =RING_RIGHT
				variableRing =RING_LEFT
			if aVarObjectYCenter >=aProObjectY2:
				processRing =RING_BOTTOM
				variableRing =RING_TOP
		if aVarObjectXCenter <= aProObjectXCenter and aVarObjectYCenter >= aProObjectYCenter:
			if aVarObjectYCenter >= aProObjectY2:
				processRing =RING_BOTTOM
				variableRing =RING_TOP
			if aVarObjectYCenter <=aProObjectY2:
				processRing =RING_LEFT
				variableRing =RING_RIGHT
		if aVarObjectXCenter <= aProObjectXCenter and aVarObjectYCenter<= aProObjectYCenter:
			if aVarObjectYCenter >= aProObjectY1:
				processRing =RING_LEFT
				variableRing =RING_RIGHT
			if aVarObjectYCenter <= aProObjectY1:
				processRing =RING_TOP
				variableRing =RING_BOTTOM



		newID = self.theLayout.getUniqueObjectID( OB_TYPE_CONNECTION )
		cmdList.append(CreateConnection( self.theLayout, newID,  processObjectID, variableObjectID, processRing, variableRing, PROCESS_TO_VARIABLE, varrefName ))
		return cmdList
		
		
		
		


	def autoShowObject( self, aFullID ):
		# return cmd or comes up with error message!
		pass
