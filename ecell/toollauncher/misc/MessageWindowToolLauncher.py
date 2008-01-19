#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
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
