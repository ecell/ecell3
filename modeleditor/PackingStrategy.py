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

		#assign process ring n var ring
		QArray = [[RING_BOTTOM,RING_TOP],[RING_LEFT,RING_RIGHT],[RING_LEFT,RING_RIGHT],[RING_TOP,RING_BOTTOM],[RING_BOTTOM,RING_TOP],[RING_RIGHT,RING_LEFT],[RING_RIGHT,RING_LEFT],[RING_TOP,RING_BOTTOM]]

		codeQ = (aVarObjectXCenter>aProObjectXCenter)*4 + (aVarObjectYCenter<aProObjectYCenter)*2 + (aVarObjectYCenter<aProObjectYCenter)
		processRing,variableRing = QArray[codeQ]
		
		return (processRing,variableRing)
		
		

	def autoShowObject( self, aFullID ):
		# return cmd or comes up with error message!
		pass
