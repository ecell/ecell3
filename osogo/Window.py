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
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from config import *

import os

import gtk
import gnome.ui
import GDK
import libglade


# ---------------------------------------------------------------
# PluginModule
#   - creates an instance of any module
# ---------------------------------------------------------------
class Window:


	# ---------------------------------------------------------------
	# Constructor
	#   - sets glade file
	#   - sets root property
	#   - call openwindow method of this class
	#
	# aGladeFile : the name glade file
	# aRoot      : root property
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aGladeFile=None, aRoot=None ):

		self.theGladeFile = aGladeFile
		self.theRoot = aRoot
		self.theTitle = "self.__name__"
		self.openWindow()

	# end of __init__


	# ---------------------------------------------------------------
	# openWindow
	#   - reads glade file
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def openWindow( self ):

		# load GLADEFILE_PATH/CLASSNAME.glade by default
		if self.theGladeFile == None:
			self.theGladeFile = GLADEFILE_PATH
			self.theGladeFile += '/' + self.__class__.__name__ + ".glade"
		else:
			if os.path.isabs( self.theGladeFile) :
				pass
			else:
				self.theGladeFile = GLADEFILE_PATH + '/' + self.theGladeFile

		if os.access( os.path.join( GLADEFILE_PATH, self.theGladeFile ), os.R_OK ):
			self.widgets = libglade.GladeXML( filename=self.theGladeFile, root=self.theRoot )
		else:
			raise IOError( "can't read %s." % self.theGladeFile )

	# end of openWindow


	# ---------------------------------------------------------------
	# addHandlers
	#   - sets some signal handers 
	#
	# handers      : signal handler dictionary
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def addHandlers( self, handlers ):

		self.widgets.signal_autoconnect( handlers )
        
	# end of addHandlers


	# ---------------------------------------------------------------
	# addHandler
	#   - sets some signal handers 
	#
	# aName       : signal name
	# aHander     : handler method 
	# *args       : argments to be set to the hander
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def addHandler( self, aName, aHandler, *args ):

		self.widgets.signal_connect( aName, aHandler, args )

	# end of addHandler


	# ---------------------------------------------------------------
	# getWidget
	#   - returns wiget specified by the key
	#   ( __getitem__ has same function )
	#
	# aKey         : widget name
	# return -> an widget
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def getWidget( self, aKey ):

		return self.widgets.get_widget( aKey )

	# end of getWidget


	# ---------------------------------------------------------------
	# __getitem__
	#   - returns wiget specified by the key
	#   ( getitemWidget has same function )
	#
	# aKey         : widget name
	# return -> an widget
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __getitem__( self, aKey ):

		return self.widgets.get_widget( aKey )

	# end of __getitem__


	# ---------------------------------------------------------------
	# editTitle
	#   - edits title of this window
	#
	# aTitle         : widget title
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def editTitle( self, aTitle ):

		self.theTitle = aTitle
		self.getWidget( self.theClassName )[ 'title' ] = self.theTitle

	# end of editTitle


# end of Window





