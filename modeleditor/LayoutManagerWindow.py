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
		self.theSelectedLayout=None

	def openWindow( self ):
		"""
		in: nothing
		returns nothing
		"""

		# superclass openwindow
		ListWindow.openWindow( self )

		self.theTreeView=ListWindow.getWidget(self,'treeview')
	
		#set up ListStore model
		self.theListStore=gtk.ListStore(gobject.TYPE_STRING)
		self.theTreeView.set_model(self.theListStore)

		# add column
		renderer=gtk.CellRendererText()
		renderer.set_property('editable','True')
                renderer.connect('edited',self.__name_edited)
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
	
	def rename (self,newName,anIter):
		if self.theSelectedLayout==newName:
			return
		else:
			aPathwayEditorList=self.theModelEditor.thePathwayEditorList
			for aPathwayEditor in aPathwayEditorList:
				if aPathwayEditor.theLayout.getName()==self.theSelectedLayout:
					self.theLayoutManager.renameLayout( self.theSelectedLayout, newName )
		self.update()
		
	#################################
	#	SIGNAL HANDLERS		#
	#################################

	def __create_layout( self, *args ):
		layoutManager = self.theModelEditor.theLayoutManager
		layoutName = layoutManager.getUniqueLayoutName()
		aCommand = CreateLayout( layoutManager, layoutName )
		self.theModelEditor.doCommandList( [ aCommand ] )


	def __delete_layout( self, *args ):
		(aListStore, anIter)=self.theListSelection.get_selected()
		if anIter:
			# delete Layout
			if self.theLayoutManager.doesLayoutExist(self.theSelectedLayout):
				self.theLayoutManager.deleteLayout(self.theSelectedLayout)

			# delete from TreeView
			aPath=aListStore.get_path(anIter)
			aListStore.remove(anIter)
			self.theListSelection.select_path(aPath)

			# if user removed the last entry, try to select the last item
			if not self.theListSelection.path_is_selected(aPath):
				aRow=aPath[0]-1
				if aRow>=0:
					self.theListSelection.select_path((aRow,))
			

	def __copy_layout( self, *args ):
		print 'Copy Button pressed'


	def __show_layout( self, *args ):
		if self.theLayoutManager.doesLayoutExist(self.theSelectedLayout):
			self.theLayoutManager.showLayout(self.theSelectedLayout)

	def __show_all_layout( self, *args ):
		aLayoutNameList=self.theLayoutManager.getLayoutNameList()
		aLayoutNameList.sort()
		for aLayoutName in aLayoutNameList:
			anIter=self.theListStore.append()
			self.theListStore.set_value(anIter,0,aLayoutName)
			if self.theSelectedLayout==aLayoutName:
				self.theListSelection.select_iter(anIter)

	def __show_selected_layout(self, *args):
		(aListStore, anIter)=self.theListSelection.get_selected()
		if anIter!=None:
			aLayoutName= aListStore.get_value(anIter,0)
			self.theSelectedLayout=aLayoutName
			# bring the associated PathwayEditorWindow to view
			aPathwayEditorList=self.theModelEditor.thePathwayEditorList
			for aPathwayEditor in aPathwayEditorList:
				if aPathwayEditor.theLayout.getName()==aLayoutName:
					if aPathwayEditor.exists():
						aPathwayEditor.present()
								

	def __name_edited(self,*args):
		'''
		args[0]=gtk.CellRenderer
		args[1]=path
		args[2]=new String
		'''
		newName=args[2]
		aPath=args[1]
		anIter=self.theListStore.get_iter_from_string(aPath)
		self.rename(newName,anIter)
