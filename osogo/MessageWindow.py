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
class MessageWindow(OsogoWindow):

	# ---------------------------------------------------------------
	# Constructor
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aMainWindow ):

		#OsogoWindow.__init__( self )
		OsogoWindow.__init__( self, aMainWindow )
		#OsogoWindow.openWindow(self)
		#self.printMessage('')
		self.theMessageBufferList=[]

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

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# save aMessage to theMessageBugger
			# --------------------------------------------------
			if OsogoWindow.getExist(self) == 0 :
				for aLine in aMessage:
					self.theMessageBufferList.append( aLine )

			# --------------------------------------------------
			# If instance of Message Window Widget has not destroyed,
			# print out aMessage on theMessageBugger
			# --------------------------------------------------
			else:
				for aLine in aMessage:
					self["message_text_box"].insert_defaults( aLine )

		# -------------------------------------------------------
		# If message is not list or touple, then print out row data.
		# -------------------------------------------------------
		else: 

			if string.find(aMessage,'\n') != 0:
				aMessage = '\n' + aMessage

			# --------------------------------------------------
			# If instance of Message Window Widget has destroyed,
			# save aMessage to theMessageBugger
			# --------------------------------------------------
			if OsogoWindow.getExist(self) == 0 :
				self.theMessageBufferList.append( aMessage )

			# --------------------------------------------------
			# If instance of Message Window Widget has not destroyed,
			# print out aMessage on theMessageBugger
			# --------------------------------------------------
			else:
				self["message_text_box"].insert_defaults( aMessage )


	# end of printMessage


	# ---------------------------------------------------------------
	# openWindow
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openWindow( self ):

		OsogoWindow.openWindow( self )
		self.printMessage( self.theMessageBufferList )
		self.theMessageBufferList = []

	# end of openWindow


if __name__ == "__main__":

	def mainLoop():
		gtk.mainloop()

	def main():
		aWindow = MessageWindow( 'MessageWindow.glade' )
		mainLoop()

	main()




