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

import gtk 
import os
from ecell.Window import *
# Constants for ConfirmWindow
OK_MODE = 0
OKCANCEL_MODE = 1

# Constans for result
OK_PRESSED = 0
CANCEL_PRESSED = -1

class LoggingPolicy(Window):
	"""This is confirm popup window class.

	OK_MODE        : The window has 'OK' button.
	OK_CANCEL_MODE : The window has 'OK' and 'Cancel' button.

	When OK is clicked, return OK_PRESSED
	When Cancel is clicked or close Window, return CANCEL_PRESSED
	"""

	# ==========================================================================
	def __init__(self, aSession, aLoggingPolicy, aTitle):
		"""Constructor
		aLoggingPolicy tuple containing logging policy
		"""

		Window.__init__(self, "LoggingPolicy.glade", "top_frame")
		self.theSession = aSession
		# Sets the return number
		self.___num = CANCEL_PRESSED

		# Create the Dialog
		self.win = gtk.Dialog("Logging Policy", None, gtk.DIALOG_MODAL)
		self.win.connect("destroy",self.destroy)
		self.win.set_title( aTitle )
		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)

		aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                              os.environ['OSOGOPATH'] + os.sep + 'ecell.png')
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                              os.environ['OSOGOPATH'] + os.sep + 'ecell32.png')
		self.win.set_icon_list(aPixbuf16, aPixbuf32)


		# Sets title
		# self.win.set_title(aTitle)
		Window.openWindow(self)
		# Sets messagd
		self.win.vbox.pack_start(self['top_frame'])
		self.win.show()

		# appends ok button
		ok_button = gtk.Button("  OK  ")
		self.win.action_area.pack_start(ok_button,gtk.FALSE,gtk.FALSE,)
		ok_button.set_flags(gtk.CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.show()
		ok_button.connect("clicked",self.oKButtonClicked)


		# appends cancel button
		cancel_button = gtk.Button(" Cancel ")
		self.win.action_area.pack_start(cancel_button,gtk.FALSE,gtk.FALSE)
		cancel_button.show()
		cancel_button.connect("clicked",self.cancelButtonClicked)

		self.__populateDialog( aLoggingPolicy )

		# add handlers
		self.addHandlers( {
			"on_space_max_toggled" : self.__buttonChosen,
			"on_space_no_limit_toggled" : self.__buttonChosen,
			"on_end_overwrite_toggled" : self.__buttonChosen,
			"on_end_throw_ex_toggled" : self.__buttonChosen,
			"on_log_by_secs_toggled" : self.__buttonChosen,
			"on_log_by_step_toggled" : self.__buttonChosen } )

		gtk.mainloop()
	# ==========================================================================
	def __populateDialog( self, aLoggingPolicy ):
		if aLoggingPolicy[0]>0:
			self['log_by_step'].set_active( gtk.TRUE )
			self['step_entry'].set_text( str(aLoggingPolicy[0] ))
			self['second_entry'].set_sensitive( gtk.FALSE )
			self['step_entry'].set_sensitive( gtk.TRUE )
		else:
			self['log_by_secs'].set_active( gtk.TRUE )
			self['second_entry'].set_text( str(aLoggingPolicy[1] ))
			self['second_entry'].set_sensitive( gtk.TRUE )
			self['step_entry'].set_sensitive( gtk.FALSE )
		if aLoggingPolicy[2]== 0:
			self['end_throw_ex'].set_active( gtk.TRUE )
		else:
			self['end_overwrite'].set_active( gtk.TRUE )

		if aLoggingPolicy[3] == 0:
			self['space_no_limit'].set_active ( gtk.TRUE )
			self['space_entry'].set_sensitive( gtk.FALSE )
		else:
			self['spac_max'].set_active( gtk.TRUE )
			self['space_entry'].set_text( str( aLoggingPolicy[3] ) )
			self['space_entry'].set_sensitive( gtk.TRUE )
			

	# ==========================================================================
	def __depopulateDialog( self ):
		aLoggingPolicy = [0,0,0,0]
		if self['log_by_step'].get_active() == gtk.TRUE:
			try:
				num = self['step_entry'].get_text()
				aLoggingPolicy[0] = int(num)
				if aLoggingPolicy[0]<0:
					a=1/0
				aLoggingPolicy[1] = 0
			except:
				self.theSession.openConfirmWindow( "Please enter valid non-negative integer for minimum step size", "Invalid number format", 0)
				return None
		else:
			try:
				aLoggingPolicy[1] = float(self['second_entry'].get_text())
				if aLoggingPolicy[1]<0:
					a=1/0
				aLoggingPolicy[0] = 0
			except:
				self.theSession.openConfirmWindow( "Please enter valid non-negative number for minimum timeinterval", "Invalid number format", 0)
				return None
		if self['end_overwrite'].get_active() == gtk.TRUE :
			aLoggingPolicy[2] = 1
		else:
			aLoggingPolicy[2] = 0
		if self['spac_max'].get_active() == gtk.TRUE:
			try:
				aLoggingPolicy[3] = int(self['space_entry'].get_text())
				if aLoggingPolicy[3]<0:
					a=1/0
			except:
				self.theSession.openConfirmWindow( "Please enter valid integer for maximum disk size", "Invalid number format", 0)
				return None
		else:
			aLoggingPolicy[3] = 0
		return aLoggingPolicy


	# ==========================================================================
	def __buttonChosen( self, *args ):
		aName = args[0].get_name()
		if aName == "log_by_secs":
			self['second_entry'].set_sensitive( gtk.TRUE )
			self['step_entry'].set_sensitive( gtk.FALSE )
		elif aName == "log_by_step":
			self['second_entry'].set_sensitive( gtk.FALSE )
			self['step_entry'].set_sensitive( gtk.TRUE )
		elif aName == "space_no_limit":
			self['space_entry'].set_sensitive( gtk.FALSE )
		elif aName == "spac_max":
			self['space_entry'].set_sensitive( gtk.TRUE )


	# ==========================================================================
	def oKButtonClicked( self, *arg ):
		"""
		If OK button clicked or the return pressed, this method is called.
		"""

		# sets the return number
		aLogPolicy = self.__depopulateDialog()
		if aLogPolicy == None:
			return
		self.___num = aLogPolicy
		self.destroy()


	# ==========================================================================
	def cancelButtonClicked( self, *arg ):
		"""
		If Cancel button clicked or the return pressed, this method is called.
		"""

		# set the return number
		self.___num = None
		self.destroy()
	

	# ==========================================================================
	def return_result( self ):
		"""Returns result
		"""

		return self.___num


	# ==========================================================================
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		self.win.hide()
		gtk.mainquit()

