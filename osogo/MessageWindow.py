#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-Cell Session Monitor package
#
#                Copyright (C) 2001-2004 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-Cell is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-Cell -- see the file COPYING.
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
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import string
from OsogoWindow import *
import gtk


class MessageWindow( Window ):
	'''
	MessageWindow
	'''

	def __init__( self ):

		Window.__init__( self, rootWidget='top_frame' )
		#		OsogoWindow.openWindow( self )
		self.isShown = False
		self.messageBuffer = gtk.TextBuffer(None)
		self.__updateEndMark()
#		self.printMessage('')


	def printMessage( self, message ):
		'''
		This method appends a message at the end of the message area.

		message can be a string, a string list or a string tuple.
		'''

		# join the strings if it is a list
		if type( message ) == list or type( message ) == tuple:  
			
			# print message list
			messageString = string.join( message )

		else:  # anything else is stringified.
			messageString = str( message )


		if len( messageString ) > 0 and messageString[0] != '\n':
			messageString = '\n' + messageString
		
		iter = self.messageBuffer.get_iter_at_mark( self.endMark )
		self.messageBuffer.insert( iter, messageString,\
					   len( messageString ) )

		if self.isShown:
			self.messageBox.scroll_to_mark( self.endMark, 0 )

	def openWindow( self ):

		self.isShown=True
		Window.openWindow( self )
		self.messageBox = self[ 'textview1' ]
		self.messageBox.set_buffer(self.messageBuffer)
		self.__updateEndMark()


	def getActualSize( self ):

		allocation = self['scrolledwindow1'].get_allocation()
		return allocation[2], allocation[3]
	
	def updateSize( self ):
		currentSize = self.getActualSize()
		self['scrolledwindow1'].set_size_request(\
			currentSize[0], currentSize[1] )

	def __updateEndMark( self ):

		endIter=self.messageBuffer.get_end_iter()
		self.endMark=self.messageBuffer.create_mark( 'EM', endIter, gtk.FALSE )


if __name__ == "__main__":

	def mainLoop():
		gtk.main()

	def main():
		aWindow = MessageWindow()
		mainLoop()

	main()




