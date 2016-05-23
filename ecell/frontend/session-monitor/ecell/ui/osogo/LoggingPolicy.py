#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import os
import gtk 

from ecell.ui.osogo.Window import *
from ecell.ui.osogo.config import *
from ecell.ui.osogo.ConfirmWindow import *

import ecell.ui.osogo.config as config

# Constants for ConfirmWindow
OK_MODE = 0
OKCANCEL_MODE = 1

# Constans for result
OK_PRESSED = 0
CANCEL_PRESSED = -1

class LoggingPolicy( Window ):
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

		Window.__init__(
			self,
			os.path.join( GLADEFILE_PATH, "LoggingPolicy.glade" ),
			"top_frame"
			)
		self.theSession = aSession
		# Sets the return number
		self.___num = None

		# Create the Dialog
		self.win = gtk.Dialog("Logging Policy", None, gtk.DIALOG_MODAL)
		self.win.connect("destroy",self.destroy)
		self.win.set_title( aTitle )
		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)

		aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, 'ecell.png') )
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, 'ecell32.png' ) )
		self.win.set_icon_list(aPixbuf16, aPixbuf32)


		# Sets title
		# self.win.set_title(aTitle)
		Window.openWindow(self)
		# Sets messagd
		self.win.vbox.pack_start(self['top_frame'])
		self.win.show()

		# appends ok button
		ok_button = gtk.Button("  OK  ")
		self.win.action_area.pack_start(ok_button,False,False,)
		ok_button.set_flags(gtk.CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.show()
		ok_button.connect("clicked",self.oKButtonClicked)


		# appends cancel button
		cancel_button = gtk.Button(" Cancel ")
		self.win.action_area.pack_start(cancel_button,False,False)
		cancel_button.show()
		cancel_button.connect("clicked",self.cancelButtonClicked)

		self.__populateDialog( aLoggingPolicy )

		# add handlers
		self.addHandlers( {
			"on_space_max_toggled" : self.__spaceLimitButtonChosen,
			"on_space_no_limit_toggled" : self.__spaceNoLimitButtonChosen,
			"on_log_by_secs_toggled" : self.__logBySecsButtonChosen,
			"on_log_by_step_toggled" : self.__logByStepButtonChosen } )


	def openConfirmWindow(self,  aMessage, aTitle, isCancel = 1 ):
		""" pops up a modal dialog window
			with aTitle (str) as its title
			and displaying aMessage as its message
			and with an OK and a Cancel button
			returns:
			True if Ok button is pressed
			False if cancel button is pressed
		"""
		aConfirmWindow = ConfirmWindow(isCancel, aMessage, aTitle )
		return aConfirmWindow.return_result() == OK_PRESSED



	def __populateDialog( self, aLoggingPolicy ):
		if aLoggingPolicy[0]>0:
			self['log_by_step'].set_active( True )
			self['step_entry'].set_text( str(aLoggingPolicy[0] ))
			self['second_entry'].set_sensitive( False )
			self['step_entry'].set_sensitive( True )
		else:
			self['log_by_secs'].set_active( True )
			self['second_entry'].set_text( str(aLoggingPolicy[1] ))
			self['second_entry'].set_sensitive( True )
			self['step_entry'].set_sensitive( False )
		if aLoggingPolicy[2]== 0:
			self['end_throw_ex'].set_active( True )
		else:
			self['end_overwrite'].set_active( True )

		if aLoggingPolicy[3] == 0:
			self['space_no_limit'].set_active ( True )
			self['space_entry'].set_sensitive( False )
		else:
			self['spac_max'].set_active( True )
			self['space_entry'].set_text( str( aLoggingPolicy[3] ) )
			self['space_entry'].set_sensitive( True )
			

	# ==========================================================================
	def __depopulateDialog( self ):
		aLoggingPolicy = [1,0,0,0]
		if self['log_by_step'].get_active() == True:
			try:
				num = self['step_entry'].get_text()
				aLoggingPolicy[0] = int(num)
				if aLoggingPolicy[0]<1:
					a=1/0
				aLoggingPolicy[1] = 0
			except:
				self.openConfirmWindow( "Please enter valid positive integer for minimum step size", "Invalid number format", 0)
				return None
		else:
			try:
				aLoggingPolicy[1] = float(self['second_entry'].get_text())
				if aLoggingPolicy[1]<0:
					a=1/0
				aLoggingPolicy[0] = 0
			except:
				self.openConfirmWindow( "Please enter valid non-negative number for minimum timeinterval", "Invalid number format", 0)
				return None
		if self['end_overwrite'].get_active() == True :
			aLoggingPolicy[2] = 1
		else:
			aLoggingPolicy[2] = 0
		if self['spac_max'].get_active() == True:
			try:
				aLoggingPolicy[3] = int(self['space_entry'].get_text())
				if aLoggingPolicy[3]<0:
					a=1/0
			except:
				self.openConfirmWindow( "Please enter valid integer for maximum disk size", "Invalid number format", 0)
				return None
		else:
			aLoggingPolicy[3] = 0
		return aLoggingPolicy


	# ==========================================================================
	def __logBySecsButtonChosen( self, *args ):
		self['second_entry'].set_sensitive( True )
		self['step_entry'].set_sensitive( False )

	def __logByStepButtonChosen( self, *args ):
		self['second_entry'].set_sensitive( False )
		self['step_entry'].set_sensitive( True )

	def __spaceNoLimitButtonChosen( self, *args ):
		self['space_entry'].set_sensitive( False )

	def __spaceLimitButtonChosen( self, *args ):
		self['space_entry'].set_sensitive( True )


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

