from Constants import *


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



	def autoConnect( self,  processObjectID, variableObjectID ):
		# return cmdlist
		pass


	def autoShowObject( self, aFullID ):
		# return cmd or comes up with error message!
		pass
