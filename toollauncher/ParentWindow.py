#!/usr/bin/env python

# This file is largely based on the OsogoWindow.py of the
# E-Cell3 Session Monitor


import os
import gtk
import gtk.gdk
import gtk.glade
from Window import *
from ToolLauncher import *

class ParentWindow(Window):
	"""ParentWindow
	- manages existance status.
	- is iconized when 'delede_event' is caught.
	"""

	# =====================================================================
	def __init__( self, aGladeFile=None ):
		"""constructor
		aGladeFile   -- a glade file name (str)
		"""

		# calls superclass's constructor
		Window.__init__( self, aGladeFile, aRoot=None )

		# initializes exist flag
		self.__theExist = False



	# ======================================================================
	def exists( self ):
		"""Returns
		TRUE: When glade file is loaded and is not deleted.
		FALSE: When glade file is not loaded yet or already deleted.
		"""

		return self.__theExist



	# =====================================================================
	def present( self ):
		"""moves this window to the top of desktop.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""
	
		# When glade file is not loaded yet or already deleted, does nothing
		# calla present() method of Window widget of this window.
		if self.exists():

			self[self.__class__.__name__].present()

	# ========================================================================
	def iconify( self ):
		"""moves this window to the taskbar.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""
	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls iconify() method of Window widget of this window.
		if self.exists():

			self[self.__class__.__name__].iconify()

	# ========================================================================
	def move( self, xpos, ypos ):
		"""moves this window on the desktop to (xpos,ypos).
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""
	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls move(x,y) method of Window widget of this window.
		if self.exists():

			self[self.__class__.__name__].move( xpos, ypos)

	# ========================================================================
	def resize( self, width, heigth ):
		"""resizes this window according to width and heigth.
		When glade file is not loaded yet or already deleted, does nothing.
		Returns None
		"""
	
		# When glade file is not loaded yet or already deleted, does nothing
		# calls resize(width,heigth) method of Window widget of this window.
		if self.exists():

			self[self.__class__.__name__].resize( width, heigth)


	# ========================================================================
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

#		self[self.__class__.__name__].iconify()

		# does not widgets
		return True


	# ========================================================================
	def openWindow( self ):
		"""overwrite super class's method
		When glade file is not loaded yet or already deleted, calls superclass's
		openWindow() method and connects 'delete_event' and self.delete() method.
		Returns None
		"""

		# when glade file is not loaded yet or already deleted.
		if self.__theExist == gtk.FALSE:

			# sets __theExist flag is TRUE
			self.__theExist = True

			# calls superclass's method 
			Window.openWindow(self)

			# connects 'delete_event' and self.delete() method.
			self[self.__class__.__name__].show_all()
			self[self.__class__.__name__].connect('delete_event',self.deleted)


	# ========================================================================
	def update( self ):
		"""
		Returns None
		"""
		pass


	# ========================================================================
	def close ( self ):
		""" destroys Widgets and sets __theExist FALSE """
		if self.exists:
			self[self.__class__.__name__].destroy()
			self.__theExist = gtk.FALSE

# end of ParentWindow
