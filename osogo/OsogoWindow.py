#!/usr/bin/env python2

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
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from config import *

import os

import gtk
#import gnome.ui
import gtk.gdk
import gtk.glade

from ecell.Window import *
from ConfirmWindow import *
from OsogoUtil import *



class OsogoWindow(Window):
	"""OsogoWindow
	- manages existance status.
	- is iconized when 'delede_event' is catched.
	"""

	def __init__( self, session = None, gladeFile=None, rootWidget=None ):
		"""constructor
		aSession  -- a reference to Session (Session)
		aGladeFile   -- a glade file name (str)
		"""

		# calls superclass's constructor
		Window.__init__( self, gladeFile, rootWidget=rootWidget )

		# saves a reference to Session
		self.theSession = session

		# initializes exist flag
		self.__theExist = False

		# set top the widget 
		self.setTopWidgetName( rootWidget )


	def setTopWidgetName( self, rootWidget ):

		if( rootWidget != None ):
			self.theTopWidget = rootWidget
		else:
			self.theTopWidget = self.__class__.__name__
		

	def exists( self ):
		"""Returns TRUE:When glade file is loaded and does not deleted.
		        FALSE:When glade file is not loaded yet or already deleted.
		"""

		return self.__theExist



	def present( self ):
		"""moves this window to the top of desktop.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""

	
		# When glade file is not loaded yet or already deleted, does nothing
		# calla present() method of Window widget of this window.
		if self.exists():

			self[self.theTopWidget].present()

	def iconify( self ):
		"""moves this window to the taskbar.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""

	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls iconify() method of Window widget of this window.
		if self.exists():

			self[self.theTopWidget].iconify()

	def move( self, xpos, ypos ):
		"""moves this window on the desktop to (xpos,ypos).
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""

	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls move(x,y) method of Window widget of this window.
		if self.exists():

			self[self.theTopWidget].move( xpos, ypos)

	def resize( self, width, heigth ):
		"""resizes this window according to width and heigth.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""

	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls resize(width,heigth) method of Window widget of this window.
		if self.exists():

			self[self.theTopWidget].resize( width, heigth)

	def deleted( self, *arg ):
		""" When 'delete_event' signal is chatcked( for example, [X] button is clicked ),
		iconize this window.
		Returns TRUE
		[Note]: 'return TRUE' means when 'delete_event' signal is checked, does not 
		        delete widgets of this class. If you'd like to delete widget, overwrite
		        this method that returns FALSE. And in the method you must set 
		        self.__theExist = FASLE.
		        example of subclass's method.

				def deleted( self, *arg ):
		            self.__theExist = FALSE
		            return FALSE

		"""

		# iconizes this window
		self.close()

		# does not widgets
		return TRUE



	def openWindow( self ):
		"""overwrite super class's method
		When glade file is not loaded yet or already deleted, calls superclass's
		openWindow() method and connects 'delete_event' and self.delete() method.
		Returns None
		"""

		# when glade file is not loaded yet or already deleted.
		if self.__theExist == gtk.FALSE:

			# sets __theExist flag is TRUE
			self.__theExist = TRUE

			# calls superclass's method 
			Window.openWindow(self)

			# connects 'delete_event' and self.delete() method.
			self[self.theTopWidget].show_all()
			self[self.theTopWidget].connect('delete_event',self.deleted)



	def update( self ):
		"""
		Returns None
		"""

		pass

	def close ( self ):
		""" destroys Widgets and sets __theExist FALSE """
		if self.exists():
			self[self.theTopWidget].destroy()
			self.__theExist = gtk.FALSE
			self.widgets = None
			self.theSession.theMainWindow.update()
#			self.theSession.updateFundamentalWindows()




