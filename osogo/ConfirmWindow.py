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
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from gtk import *

# --------------------------------------------------------
# This is confirm popup window class.
#
# mode 0 : The window has 'OK' button.
# mode 1 : The window has 'OK' and 'Cancel' button.
#
# If OK is clicked, return 0
# If Cancel is clicked or close Window, return -1
#
# --------------------------------------------------------
class ConfirmWindow(GtkDialog):

	# ----------------------------------------------------
	# Constructor
	# aMode : mode number that is 0(OK) or 1(OK and Cancel).
	# aMessage : the message that is displayed in the center
	#            of this window
	# aTitle : the title of this window
	# ----------------------------------------------------
	def __init__(self, aMode, aMessage, aTitle=None  ):

		# Sets the return number
		self._num = -1

		# Create the Dialog
		self.win = GtkDialog()
		self.win.connect("destroy",self.hide)
		#self.win.connect("destroy",self.quit)

		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(WIN_POS_MOUSE)
		self.win.show()

		# Sets title
		if aTitle != None:
			self.win.set_title(aTitle)

		# Sets message
		aMessageLabel = GtkLabel(aMessage)
		self.win.vbox.pack_start(aMessageLabel)
		aMessageLabel.show()
	
		ok_button = GtkButton("  OK  ")
		self.win.action_area.pack_start(ok_button,FALSE,FALSE,)
		ok_button.set_flags(CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.show()
		ok_button.connect("clicked",self.OKButtonClicked)

		if aMode == 0:
			pass
		else:
			cancel_button = GtkButton(" Cancel ")
			self.win.action_area.pack_start(cancel_button,FALSE,FALSE)
			cancel_button.show()
			cancel_button.connect("clicked",self.CancelButtonClicked)	

		mainloop()

	# end of __init__

	# ----------------------------------------------------
	# If OK button clicked or the return pressed, this
	# method is called.
	# ----------------------------------------------------
	def OKButtonClicked( self, *obj ):
		# set the return number
		self._num = 0
		self.quit()

	# end of OKButtonClicked

	# ----------------------------------------------------
	# If Cancel button clicked or the return pressed, this
	# method is called.
	# ----------------------------------------------------
	def CancelButtonClicked( self, *obj ):
		# set the return number
		self._num = -1
		self.quit()
	
	# end of CancelButtonClicked

	# ----------------------------------------------------
	# Override the method of return_result
	# ----------------------------------------------------
	def return_result( self ):
		return self._num

	# end of return_result

	# ----------------------------------------------------
	# quit
	# ----------------------------------------------------
	def quit( self, *obj ):
		self.win.hide()
		mainquit()

	# end of quit

# ----------------------------------------------------
# Test code
# ----------------------------------------------------
if __name__=="__main__":
	c = ConfirmWindow(0,'hoge\n')
	print c.return_result()


