#!/usr/bin/env python2

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
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from config import *

import os

import gtk
import gnome.ui
#import GDK
import gtk.gdk
#import libglade
import gtk.glade

from ecell.Window import *
from ConfirmWindow import *


# ---------------------------------------------------------------
# OsogoWindow -> Window
#   - has existance status of window widget 
#     the member theExist is 0 -> Window widget doesn't exist.
#                            1 -> Window widget exists.
# ---------------------------------------------------------------
class OsogoWindow(Window):

	# ---------------------------------------------------------------
	# Constructor
	#   - initialize exist parameter
	#
	# return -> None
	# ---------------------------------------------------------------
	def __init__( self, aMainWindow, aGladeFile=None, aRoot=None ):

		self.theMainWindow = aMainWindow
		self.theExist = gtk.FALSE
		self.isShown = gtk.FALSE
		Window.__init__( self, aGladeFile, aRoot )

	# end of __init__


	# ---------------------------------------------------------------
	# getExist
	#  
	# return -> exist status  
	# ---------------------------------------------------------------
	def getExist( self ):

		return self.theExist

	# end of getExist


	# ---------------------------------------------------------------
	# getIsShown
	#  
	# return -> shown status  
	# ---------------------------------------------------------------
	def getIsShown( self ):

		return self.isShown

	# end of getIsShown


	# ---------------------------------------------------------------
	# destroyWindow
	#  
	# *objects : dammy objects
	#
	# return -> exist status  
	# ---------------------------------------------------------------
	def destroyWindow( self, *objects ):

		self.theExist = gtk.FALSE
		self.isShown = gtk.FALSE
		#try:
		self.theMainWindow.updateFundamentalWindows()
		#except:
		#	pass

	# end of destroyWindow


	# ---------------------------------------------------------------
	# openWindow
	#  
	# return -> None
	# ---------------------------------------------------------------
	def openWindow( self ):


		# --------------------------------------------------
		# If instance of Window Widget has destroyed,
		# creates instance of Window Widget.
		# --------------------------------------------------
		if self.theExist == gtk.FALSE:
			Window.openWindow(self)

		# --------------------------------------------------
		# If instance of Message Window Widget has destroyed,
		# calls the show method of Window Widget.
		# --------------------------------------------------
		self[self.__class__.__name__].show_all()

		# sets signalhander
		self[self.__class__.__name__].connect('destroy',self.destroyWindow)

		# sets exist status 'exists'
		self.theExist = gtk.TRUE
		self.isShown = gtk.TRUE

	# end of openWindow


	def update( self ):
		pass

# end of OsogoWindow


