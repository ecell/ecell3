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

from OsogoWindow import *
from OsogoUtil import *

import gtk
from ecell.ecssupport import *
import gobject

import string
import copy

class PluginInstanceSelection(OsogoWindow):
	"""PluginInstanceSelection
	"""

	def __init__( self, aSession, anEntityListWindow ):
		"""Constructor
		aSession         ---   a reference to Session (Session)
		anEntityListWindow  ---   a reference to EntityListWindow (EntityListWindow)
		"""

		# calls superclass's constructor 
		OsogoWindow.__init__( self, aSession )

		self.theEntityListWindow = anEntityListWindow
		self.thePluginManager = aSession.thePluginManager


	# ====================================================================
	def openWindow( self ):

		# calls superclass's openWindow
		OsogoWindow.openWindow( self )

		# add handers
		self.addHandlers( { 
		'on_ok_button_plugin_selection_clicked'      : \
		self.theEntityListWindow.appendData,\
		'on_cancel_button_plugin_selection_clicked'  : \
		self.theEntityListWindow.closePluginInstanceSelectionWindow,\
		                   } )
		self.__initializePluginInstanceSelectonWindow()
		self[self.__class__.__name__].connect('delete_event', self.theEntityListWindow.closePluginInstanceSelectionWindow )
		self.show_all()

	# ====================================================================
	def deleted( self, *arg ):
		self['PluginInstanceSelection'].hide_all()
		#self.theEntityListWindow.closePluginInstanceSelectionWindow()
		return FALSE

	# ====================================================================
	def update( self ):
		"""updates list 
		Returns None
		"""

		self.thePluginInstanceListStore.clear()
		aPluginInstanceList = self.thePluginManager.thePluginTitleDict.keys()
		for aPluginInstance in aPluginInstanceList:
			if aPluginInstance.theViewType == MULTIPLE:
				aPluginInstanceTitle = self.thePluginManager.thePluginTitleDict[aPluginInstance]
				iter = self.thePluginInstanceListStore.append()
				self.thePluginInstanceListStore.set_value( iter, 0, aPluginInstanceTitle )
				self.thePluginInstanceListStore.set_data( aPluginInstanceTitle, aPluginInstanceTitle )

	
	# ====================================================================
	def __initializePluginInstanceSelectonWindow( self ):
		"""initializes PluginInstanceSelectionWindow
		Returns None
		"""

		column = gtk.TreeViewColumn( 'Plugin List', gtk.CellRendererText(), text=0 )
		self['plugin_tree'].append_column(column)
		self.thePluginInstanceListStore=gtk.ListStore( gobject.TYPE_STRING )
		self['plugin_tree'].get_selection().set_mode( gtk.SELECTION_MULTIPLE )
		self['plugin_tree'].set_model( self.thePluginInstanceListStore )
		column = gtk.TreeViewColumn( 'Plugin List', gtk.CellRendererText(), text=0 )

	# ========================================================================
	def plugin_select_func(self,tree,path,iter):
		key=self.thePluginInstanceListStore.get_value(iter,0)
		aTitle = self.thePluginInstanceListStore.get_data( key )
		self.theEntityListWindow.theSelectedPluginInstanceList.append( aTitle )



