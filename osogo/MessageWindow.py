#!/usr/bin/env python


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
#             'Yuusuke Saito'
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

import string
from OsogoWindow import *
from gtk import *

# ---------------------------------------------------------------
# MessageWindow -> OsogoWindow
#   - manages MessageWindow
# ---------------------------------------------------------------
class MessageWindow:

	# ---------------------------------------------------------------
	# Constructor
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, theMessageBox ):

		#OsogoWindow.__init__( self )
#		OsogoWindow.__init__( self, aMainWindow )
		#OsogoWindow.openWindow(self)
		#self.printMessage('')
		self.theMessageBufferList=gtk.TextBuffer(None)
		self.theMessageBox=theMessageBox			
	# end of __init__


	# ---------------------------------------------------------------
	# PrintMessage
	#
	# aMessage(string or list or touple)
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def printMessage( self, aMessage ):

		# -------------------------------------------------------
		# If messge is list or touple, then print out each line.
		# -------------------------------------------------------
		if type(aMessage) == type([]) or type(aMessage) == type(()) :  
			
			if len(aMessage)>0:
				if string.find(aMessage[0],'\n') != 0:
					aMessage = '\n' + aMessage[0]

			for aLine in aMessage:
				aString = str( aMessage )
				self.theMessageBufferList.insert_at_cursor( aString, len(aString) )


		# -------------------------------------------------------
		# If message is not list or touple, then print out row data.
		# -------------------------------------------------------
		else: 
			aString = str( aMessage )
			if string.find(aString,'\n') != 0:
				aString = '\n' + aString
			self.theMessageBufferList.insert_at_cursor( aString ,len(aString) )
		
		self.theMessageBox.scroll_to_mark(self.EndMark,0)


	# end of printMessage


	# ---------------------------------------------------------------
	# openWindow
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openWindow( self ):

		self.isShown=gtk.TRUE
#		OsogoWindow.openWindow( self )
#		self.printMessage( self.theMessageBufferList )
#		self.theMessageBufferList = []
		self.theMessageBox.set_buffer(self.theMessageBufferList)
		EndIter=self.theMessageBufferList.get_end_iter()
		self.EndMark=self.theMessageBufferList.create_mark('EM',EndIter,gtk.FALSE)
		
	# end of openWindow


if __name__ == "__main__":

	def mainLoop():
		gtk.mainloop()

	def main():
		aWindow = MessageWindow( 'MessageWindow.glade' )
		mainLoop()

	main()




