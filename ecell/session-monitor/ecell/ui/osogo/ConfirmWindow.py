#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

import gtk 
import os
import config

# Constants for ConfirmWindow
OK_MODE = 0
OKCANCEL_MODE = 1

# Constans for result
OK_PRESSED = 0
CANCEL_PRESSED = -1

class ConfirmWindow(gtk.Dialog):
	"""This is confirm popup window class.

	OK_MODE        : The window has 'OK' button.
	OK_CANCEL_MODE : The window has 'OK' and 'Cancel' button.

	When OK is clicked, return OK_PRESSED
	When Cancel is clicked or close Window, return CANCEL_PRESSED
	"""

	def __init__( self, aMode, aMessage, aTitle='Confirm' ):
		"""Constructor
		aMode    ---  mode number that is 0(OK) or 1(OK and Cancel).
		aMessage ---  the message that is displayed in the center
		              of this window
		aTitle   ---  the title of this window
		"""

		# Sets the return number
		self.___num = CANCEL_PRESSED

		# Create the Dialog
		gtk.Dialog.__init__( self, aTitle, None, gtk.DIALOG_MODAL )

		# Sets size and position
		self.set_border_width(2)
		self.set_default_size(300,75)
		self.set_position(gtk.WIN_POS_MOUSE)

		aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.glade_dir, 'ecell.png') )
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.glade_dir, 'ecell32.png') )
		self.set_icon_list(aPixbuf16, aPixbuf32)		

		# Sets title
		# self.set_title(aTitle)

		# Sets message
		aMessage = '\n' + aMessage + '\n'
		aMessageLabel = gtk.Label(aMessage)
		self.vbox.pack_start(aMessageLabel)
	
		# appends ok button
		ok_button = gtk.Button("  OK  ")
		self.action_area.pack_start(ok_button,False,False,)
		ok_button.set_flags(gtk.CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.connect("clicked",self.oKButtonClicked)

		# when ok mode 
		if aMode == OK_MODE:
			pass
		# when ok cancel mode 
		else:
			# appends cancel button
			cancel_button = gtk.Button(" Cancel ")
			self.action_area.pack_start(cancel_button,False,False)
			cancel_button.connect("clicked",self.cancelButtonClicked)	

	def oKButtonClicked( self, *arg ):
		"""If OK button clicked or the return pressed, this method is called.
		"""

		# sets the return number
		self.___num = OK_PRESSED
		self.destroy()

	def cancelButtonClicked( self, *arg ):
		"""If Cancel button clicked or the return pressed, this method is called.
		"""

		# set the return number
		self.___num = CANCEL_PRESSED
		self.destroy()

	def getResult( self ):
		"""Returns result
		"""

		return self.___num
