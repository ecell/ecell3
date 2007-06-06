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
#!/usr/bin/python
#
# ToolLauncherOkCancel.py  - E-Cell3 Tool Launcher OkCancel Window
#

from ParentWindow import *
import sys

try:
	import gtk
	import os
except KeyboardInterrupt:
	sys.exit(1)

class ToolLauncherOkCancel( ParentWindow ):

	def __init__( self, obj ):
		"""Constructor 
		- calls parent class's constructor
		- calls openWindow
		"""
		ParentWindow.__init__( self, 'ToolLauncherOkCancel.glade' )
		self.theObj = obj
	# end of __init__


	def openWindow( self, msg ):
		"""overwrite parent class' method
		Returns None
		"""
		ParentWindow.openWindow( self )
		self.theHandlerMap =  {
			'on_ok_button_clicked'                  : self.onOk ,
			'on_cancel_button_clicked'              : self.onCancel ,
		}
		self.addHandlers( self.theHandlerMap )
                self.setIconList(
			os.environ['TLPATH'] + os.sep + "toollauncher.png",
			os.environ['TLPATH'] + os.sep + "toollauncher32.png")
                self.__update( msg )
	# end of openWindow


	def __update( self, msg ):
	    	"""update the checkboxes and entry boxes with the preferences from ToolLauncher
		"""
		self['errormsg'].set_text( msg )
		self.update()
	# end of __update


	def onOk( self, *arg ):
		"""when OK button is clicked
		"""
		self.theObj.onOk()
		self.close()
		return True
	# end of onOk


	def onCancel( self, *arg ):
		"""when Cancel button is clicked
		"""
#		self.theObj.onCancel()
		self.close()
#		return False
	# end of onCancel
# end of ToolLauncherOkCancel

