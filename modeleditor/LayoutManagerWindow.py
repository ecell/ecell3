#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#		This file is part of E-CELL Model Editor package
#
#				Copyright (C) 1996-2003 Keio University
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
#'Design: Dini Karnaga, Sylvia Tarigan, Thaw Tint <polytech@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Dini Karnaga, Sylvia Tarigan, Thaw Tint' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


import gtk
import gobject
from ModelEditor import *
from ListWindow import *
import os
import os.path
import string
from LayoutCommand import *

class LayoutManagerWindow( ListWindow ):


	def __init__( self, theModelEditor ):
	             
		"""
		in: ModelEditor theModelEditor
		returns nothing
		"""
		self.theModelEditor=theModelEditor
		# init superclass
		ListWindow.__init__( self, theModelEditor )

		# the variable
		self.theLayoutManager=self.theModelEditor.theLayoutManager
		self.theTreeView=None 

	def openWindow( self ):
		"""
		in: nothing
		returns nothing
		"""

		# superclass openwindow
		ListWindow.openWindow( self )

		self.theTreeView=ListWindow.getWidget(self,'treeview')
	
		#set up ListStore
		self.theListStore=gtk.ListStore(gobject.TYPE_STRING)
		self.theTreeView.set_model(self.theListStore)
		renderer=gtk.CellRendererText()
		column = gtk.TreeViewColumn("Layout Name", renderer, text=0)
		self.theTreeView.append_column(column)

		# set up the variables
		self.theListSelection = self.theTreeView.get_selection()
		self.theListSelection.connect("changed", self.__show_selected_layout)

		# add signal handlers
		self.addHandlers({ 
				'on_CreateButton_clicked' : self.__create_layout,\
				'on_DeleteButton_clicked' : self.__delete_layout,\
				'on_CopyButton_clicked' : self.__copy_layout,\
				'on_ShowButton_clicked' : self.__show_layout,\
				 })
		
		# show list of available layout, if any
		self.__show_all_layout()
	
	
	def move( self, xpos, ypos ):
			ListWindow.move(self,xpos,ypos)
	
	def deleted( self, *args ):
		ListWindow.deleted( self, args )
		self.theModelEditor.toggleOpenLayoutWindow(False)
	
	def update ( self ):
		self.theListStore.clear()
		self.__show_all_layout()

	def __create_layout( self, *args ):
		layoutManager = self.theModelEditor.theLayoutManager
		layoutName = layoutManager.getUniqueLayoutName()
		aCommand = CreateLayout( layoutManager, layoutName )
		self.theModelEditor.doCommandList( [ aCommand ] )


	def __delete_layout( self, *args ):
		print 'Delete Button pressed'

	def __copy_layout( self, *args ):
		print 'Copy Button pressed'

	def __show_layout( self, *args ):
		print 'Show Button pressed' 

	def __show_all_layout( self, *args ):
		aLayoutNameList=self.theLayoutManager.getLayoutNameList()
		aLayoutNameList.sort()
		for aLayoutName in aLayoutNameList:
			anIter=self.theListStore.append()
			self.theListStore.set_value(anIter,0,aLayoutName)

	def __show_selected_layout(self, *args):
		(aListStore, anIter)=self.theListSelection.get_selected()
		aLayoutName= aListStore.get_value(anIter,0,)
		print "You selected %s"%aLayoutName
		
