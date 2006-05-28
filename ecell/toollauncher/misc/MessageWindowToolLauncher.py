#!/usr/bin/python
#
# MessageWindowToolLauncher.py
#

from MessageWindow import *
from gtk import *

class MessageWindowToolLauncher(MessageWindow):

	# ==========================================================================
	def __init__( self ):

		"""Constructor 
		- calls parent class's constructor
		"""
		MessageWindow.__init__( self )

	# enf of __init__


	# ==========================================================================
	def refreshMessage( self ):

		self.refreshMessageMsg( )

	# end of refreshMessage


	# ==========================================================================
	def getMessage( self ):

		StrIter=self.theMessageBufferList.get_start_iter()
		EndIter=self.theMessageBufferList.get_end_iter()
		return self.theMessageBufferList.get_text(StrIter , EndIter , True)

	# end of getMessage


	# ==========================================================================
	def deleteMessage( self ):

		strIter=self.theMessageBufferList.get_start_iter()
		endIter=self.theMessageBufferList.get_end_iter()
		self.theMessageBufferList.delete(strIter, endIter)

	# enf of deleteMessage
