class Error(Exception):
	"""Base class for exceptions in this module."""
   	pass

class ClassNotExistError(Error):
	"""Exception raised for errors in loading .desc files
	Attributes:
        class -- the class without .desc file
	"""

	def __init__(self,aClass): 
        	self.args = "No .desc file for %s"%aClass

	#def __str__(self):
	#	return repr(self.message)

