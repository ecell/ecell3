
class CommandQueue:


	def __init__(self, aLength ):
		self.__theLength = aLength
		self.__theCommandQueue = []
		self.__thePointer = 0

	def push( self, aCommandList):
		if self.isNext():
			self.__deleteForward()
		self.__theCommandQueue.append( aCommandList )
		self.__thePointer += 1
		if len(self.__theCommandQueue) > self.__theLength:
			self.__theCommandQueue.__delitem__(0)
			self.__thePointer -= 1
			


	def moveback( self ):
		if self.isPrevious():
			self.__thePointer-=1
			return self.__theCommandQueue[ self.__thePointer ]
		raise Exception("No way to move back back in CommandQueue!")


	def moveforward( self):
		if self.isNext():
			self.__thePointer+=1
			return self.__theCommandQueue[ self.__thePointer -1 ]
		raise Exception("No way to move forward in CommandQueue")


	def __deleteForward( self ):
		while self.isNext():
			self.__theCommandQueue.pop()


	def isPrevious( self ):
		return self.__thePointer > 0


	def isNext( self ):
		return self.__thePointer < len( self.__theCommandQueue)


