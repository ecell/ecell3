#!/usr/bin/env python

from OsogoWindow import *
from PluginInstanceSelection import *

import gtk
from ecell.ecssupport import *
import gobject

import operator
import string
import re
import copy

# This parameter should be set by setting file.
DEFAULT_WINDOW = 'TracerWindow'

DEFAULT_VARIABLE_PROPERTY = 'Value'
DEFAULT_PROCESS_PROPERTY = 'Activity'


class EntityListWindow(OsogoWindow):
	"""EntityListWindow
	"""

	VARIABLE_COLUMN_LIST= ( 'ID',\
				'Value',\
				'Classname',\
				'Path' )

	PROCESS_COLUMN_LIST=( 'ID',\
			      'Activity',\
			      'Classname',\
			      'Path' )

	def __init__( self, aSession ):
		'''Constructor
		aSession   --   a reference to GtkSessionMonitor
		'''
		
		# call superclass's constructor 
		OsogoWindow.__init__( self, aSession )
		
		# initialize parameters
		self.theSelectedFullPNList = []

		self.searchString = ''
		
		# fix me
		if( self.theSession != None ):
			self.thePluginManager = aSession.thePluginManager
			
		self.thePluginInstanceSelection = None
			
		self.theSelectedSystemIter = None

	def openWindow( self ):

		# call superclass's openWindow
		OsogoWindow.openWindow( self )
		
		
		# add handers
		self.addHandlers( { 
			# system tree
			'on_system_tree_cursor_changed' :\
			self.updateSystemSelection,\
			'on_system_tree_button_press_event' : self.popupMenu,\
			# entity list
#			'on_process_tree_cursor_changed': self.selectProcess,\
#			'on_variable_tree_cursor_changed':self.selectVariable,\
			'on_view_button_clicked': self.createPluginWindow,\
			'on_variable_tree_button_press_event': self.popupMenu,\
			'on_process_tree_button_press_event': self.popupMenu,\
			# search 
			'on_search_button_released': self.pushSearchButton,\
			'on_search_entry_key_press_event':\
			self.keypressOnSearchEntry,\
			} )


		self.theLastSelectWindow = None

		
		self.systemTree   = self['system_tree']
		self.processTree  = self['process_tree']
		self.variableTree = self['variable_tree']
		
		# --------------------------------------------
		# initialize components
		# --------------------------------------------

		if( self.theSession == None ):
			self['search_button'].set_sensitive(0)
			self['view_button'].set_sensitive(0)
			self['search_entry'].set_sensitive(0)
			self['plugin_optionmenu'].set_sensitive(0)
		else:

			self.__initializeSystemTree()
			self.__initializeProcessTree()
			self.__initializeVariableTree()

			self.__initializePluginWindowOptionMenu()
			self.__initializePropertyWindow()
			self.__initializePopupMenu()


			# --------------------------------------------
			# initialize system tree
			# --------------------------------------------
			# set up system tree
			self.theSysTreeStore.clear()
			
			# create route ID
			aRootSystemFullID = createFullID( 'System::/' )
			self.constructTree( None, aRootSystemFullID )
			self.update()

			# --------------------------------------------
			# initialize buffer
			# --------------------------------------------
			self.theSelectedEntityList = []
			self.theSelectedPluginInstanceList = []
			
			# --------------------------------------------
			# initialize Add to Board button
			# --------------------------------------------
		        #self.checkBoardExists()
		        self.CloseOrder = False

	def setSession( self, aSession ):
		self.theSession = aSession
		self.thePluginManager = aSession.thePluginManager

		self['search_button'].set_sensitive(1)
		self['view_button'].set_sensitive(1)
		self['search_entry'].set_sensitive(1)
		self['plugin_optionmenu'].set_sensitive(1)

		self.__initializeSystemTree()
		self.__initializeProcessTree()
		self.__initializeVariableTree()
		
		self.__initializePluginWindowOptionMenu()
		self.__initializePropertyWindow()
		self.__initializePopupMenu()

		
		# --------------------------------------------
		# initialize system tree
		# --------------------------------------------
		# set up system tree
		self.theSysTreeStore.clear()
		
		# create route ID
		aRootSystemFullID = createFullID( 'System::/' )
		self.constructTree( None, aRootSystemFullID )
		self.update()
		
		# --------------------------------------------
		# initialize buffer
		# --------------------------------------------
		self.theSelectedEntityList = []
		self.theSelectedPluginInstanceList = []
			
		# --------------------------------------------
		# initialize Add to Board buttons
		# --------------------------------------------
		#self.checkBoardExists()
		self.CloseOrder = False
			
	def checkBoardExists( self ):
		if self.theSession.getWindow('BoardWindow').exists():
			self['add_to_board'].set_sensitive(TRUE)
		else:
			self['add_to_board'].set_sensitive(FALSE)
		

	def deleted( self, *arg ):
		self.close()


	def close( self ):
		if self.CloseOrder:
			return
		self.CloseOrder = True

		if self.thePluginInstanceSelection != None:
			self.thePluginInstanceSelection.deleted()
			self.thePluginInstanceSelection = None
		self.theSession.deleteEntityListWindow( self )
		OsogoWindow.close(self)
	
	def deletePluginInstanceSelection( self, *arg ):
		"""sets 'delete_event' as 'hide_event'
		"""

		# hide this window
		self['PluginInstanceSelection'].hide_all()

		# set 'delete_event' uneffective
		return TRUE

	def __initializeSystemTree( self ):
		"""initialize SystemTree
		"""

		self.theSysTreeStore=gtk.TreeStore( gobject.TYPE_STRING )
		self.systemTree.set_model(self.theSysTreeStore)
		column=gtk.TreeViewColumn( 'System Tree',
					   gtk.CellRendererText(),
					   text=0 )
		column.set_visible( gtk.TRUE )
		self.systemTree.append_column(column)


	def __initializeProcessTree( self ):
		"""initialize ProcessTree
		"""

		columnTypeList = []

		for i in range( len( self.PROCESS_COLUMN_LIST ) ):
			title = self.PROCESS_COLUMN_LIST[i]
			
			column = gtk.TreeViewColumn( title,
						     gtk.CellRendererText(),
						     text=i )
			column.set_reorderable( True )
			column.set_sort_column_id( i )
			self.processTree.append_column( column )
			columnTypeList.append( gobject.TYPE_STRING )


		selection = self.processTree.get_selection()
		selection.set_mode(\
			gtk.SELECTION_MULTIPLE )
		selection.connect( 'changed',\
				   self.selectProcess )
		self.processTree.set_model( gtk.ListStore( *columnTypeList ) )


	def __initializeVariableTree( self ):
		"""initializes VariableTree
		"""

		columnTypeList = []

		for i in range( len( EntityListWindow.VARIABLE_COLUMN_LIST ) ):
			title = EntityListWindow.VARIABLE_COLUMN_LIST[i]
			
			column = gtk.TreeViewColumn( title,
						     gtk.CellRendererText(),
						     text=i )
			column.set_reorderable( True )
			column.set_sort_column_id( i )
			self.variableTree.append_column( column )
			columnTypeList.append( gobject.TYPE_STRING )

		selection = self.variableTree.get_selection()
		selection.set_mode(\
			gtk.SELECTION_MULTIPLE )
		selection.connect( 'changed',\
				   self.selectVariable )
		self.variableTree.set_model( gtk.ListStore( *columnTypeList ) )

		


	def __initializePluginWindowOptionMenu( self ):
		"""initializes PluginWindowOptionMenu
		"""

		aPluginWindowNameList = []
		aMenu = gtk.Menu()

		for aPluginWindowName in self.thePluginManager.thePluginMap.keys():

			aButton = gtk.Button()
			aMenuItem = gtk.MenuItem(aPluginWindowName)

			if aPluginWindowName == DEFAULT_WINDOW:
				aMenu.prepend( aMenuItem )
			else:
				aMenu.append( aMenuItem )

		self['plugin_optionmenu'].set_menu(aMenu)
		self['plugin_optionmenu'].show_all()

		# set default menu


	def __initializePropertyWindow( self ):

		self.thePropertyWindow= \
		self.thePluginManager.createInstance( 'PropertyWindow',\
						      [(SYSTEM, '', '/', '')],\
						      rootWidget= 'top_frame',\
						      parent= self ) 
		self.thePropertyWindow.setStatusBar( self['statusbar'] )

		aPropertyWindowTopVBox = self.thePropertyWindow['top_frame']
		self['property_area'].add( aPropertyWindowTopVBox )
		self.thePropertyWindow.setParent( self )



	def __initializePopupMenu( self ):
		"""Initialize popup menu
		Returns None
		[Note]:In this method, only 'PluginWindow type' menus, 'Create 
		       Logger' menu and 'Add to Board' menu are created. 
		       The menus of PluginWindow instances are appended
		       dinamically in self.popupMenu() method.
		"""

		# ------------------------------------------
		# menus for PluginWindow
		# ------------------------------------------

		# creaets menus of PluginWindow
		for aPluginWindowType in self.thePluginManager.thePluginMap.keys(): 
			aMenuItem = gtk.MenuItem( aPluginWindowType )
			aMenuItem.connect('activate', self.createPluginWindow )
			aMenuItem.set_name( aPluginWindowType )
			if aPluginWindowType == DEFAULT_WINDOW:
				self['EntityPopupMenu'].prepend( aMenuItem )
			else:
				self['EntityPopupMenu'].append( aMenuItem )

		# appends separator
		self['EntityPopupMenu'].append( gtk.MenuItem() )

		# ------------------------------------------
		# menus for Logger
		# ------------------------------------------
		# creates menu of Logger
		aLogMenuString = "Create Logger"
		aMenuItem = gtk.MenuItem( aLogMenuString )
		aMenuItem.connect('activate', self.createLogger )
		aMenuItem.set_name( aLogMenuString )
		self['EntityPopupMenu'].append( aMenuItem )

		# appends separator
		self['EntityPopupMenu'].append( gtk.MenuItem() )

		# ------------------------------------------
		# menus for Bord
		# ------------------------------------------
		# creates menu of Board
		aSubMenu = gtk.Menu()

		for aPluginWindowType in self.thePluginManager.thePluginMap.keys(): 
			aMenuItem = gtk.MenuItem( aPluginWindowType )
			aMenuItem.connect('activate', self.addToBoard )
			aMenuItem.set_name( aPluginWindowType )
			if aPluginWindowType == DEFAULT_WINDOW:
				aSubMenu.prepend( aMenuItem )
			else:
				aSubMenu.append( aMenuItem )

		aMenuString = "Add to Board"
		aMenuItem = gtk.MenuItem( aMenuString )
		aMenuItem.set_name( aLogMenuString )
		aMenuItem.set_submenu( aSubMenu )
		self['EntityPopupMenu'].append( aMenuItem )
		self.theBoardMenu = aMenuItem

		# appends separator
		self['EntityPopupMenu'].append( gtk.MenuItem() )

		# ------------------------------------------
		# menus for submenu
		# ------------------------------------------
		self.thePopupSubMenu = None  




	def __openPluginInstanceSelectionWindow( self, *arg ):
		"""open PluginInstanceSelectionWindow
		Returns None
		"""

		if self.thePluginInstanceSelection != None:
			self.thePluginInstanceSelection.present()

		else:

			self.thePluginInstanceSelection = \
			PluginInstanceSelection( self.theSession, self )
			self.thePluginInstanceSelection.openWindow()

			# updates list of PluginInstance
			self.thePluginInstanceSelection.update()




	def __updatePluginInstanceSelectionWindow2( self ):
		"""updates list of PluginInstanceSelectionWindow
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


	def closePluginInstanceSelectionWindow( self, *arg ):
		"""closes PluginInstanceSelectionWindow
		Returns None
		"""

		if self.thePluginInstanceSelection != None:
			#self.thePluginInstanceSelection['PluginInstanceSelection'].hide_all()
			self.thePluginInstanceSelection.deleted()
			self.thePluginInstanceSelection = None



	def popupMenu( self, aWidget, anEvent ):
		"""displays popup menu only when right button is pressed.
		aWidget   --  EntityListWindow
		anEvent   --  an event
		Returns None
		[Note]:creates and adds submenu that includes menus of PluginWindow instances
		"""
		# When right button is pressed
		if anEvent.type == gtk.gdk._2BUTTON_PRESS:

			aSelectedRawFullPNList = self.__getSelectedRawFullPNList()
			aPluginWindowType = self['plugin_optionmenu'].get_children()[0].get()

			# When no FullPN is selected, displays error message.
			if aSelectedRawFullPNList  == None or len( aSelectedRawFullPNList ) == 0:

				aMessage = 'No entity is selected.'
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				self.thePropertyWindow.showMessageOnStatusBar(aMessage)
				return FALSE

			self.thePropertyWindow.setRawFullPNList( aSelectedRawFullPNList )
			self.thePluginManager.createInstance( aPluginWindowType, self.thePropertyWindow.theFullPNList() )

			

		# When right button is pressed
		if anEvent.type == gtk.gdk.BUTTON_PRESS and anEvent.button == 3:

			if self.theSession.getWindow('BoardWindow').exists():
				self.theBoardMenu.set_sensitive(TRUE)
			else:
				self.theBoardMenu.set_sensitive(FALSE)

			# removes previous sub menu
			# When PopupMenu was displayed last time without PluginWindows'
			# menus, the buffer (self.thePopupSubMenu) is None.
			if self.thePopupSubMenu != None:
				self['EntityPopupMenu'].remove( self.thePopupSubMenu )

			if len(self.thePluginManager.theInstanceList)!=0:

				# creates submenu
				aSubMenu = gtk.Menu()

				# creaets menus of PluginWindow instances
				aMenuItemFlag = FALSE
				for aPluginInstance in self.thePluginManager.theInstanceList: 
		
					if aPluginInstance.theViewType == MULTIPLE:
						aTitle = aPluginInstance.getTitle()
						aMenuItem = gtk.MenuItem( aTitle )
						aMenuItem.connect('activate', self.appendData )
						aMenuItem.set_name( aTitle )
						aSubMenu.append( aMenuItem )
						aMenuItemFlag = TRUE

				if aMenuItemFlag == TRUE:
					# creates parent MenuItem attached created submenu.
					aMenuString = "Append data to"
					aMenuItem = gtk.MenuItem( aMenuString )
					aMenuItem.set_submenu( aSubMenu )

					# appends parent MenuItem to PopupMenu
					self['EntityPopupMenu'].append( aMenuItem )

					# saves this submenu set to buffer (self.thePopupSubMenu)
					self.thePopupSubMenu = aMenuItem


			# displays all items on PopupMenu
			self['EntityPopupMenu'].show_all() 

			# displays popup menu
			self['EntityPopupMenu'].popup(None,None,None,anEvent.button,anEvent.time)



	def update( self ):
		"""overwrite superclass's method
		updates this window and property window
		Returns None
		"""
		
		# updates this window
		OsogoWindow.update(self)

		# updates property window
		self.thePropertyWindow.update()

		# update PluginInstanceSelectionWindow
		if self.thePluginInstanceSelection != None:
			self.thePluginInstanceSelection.update()

		self.updateLists()


	def constructTree( self, aParentTree, aSystemFullID ):

		aNewlabel = aSystemFullID[ID] 

		iter  = self.theSysTreeStore.append( aParentTree )
		self.theSysTreeStore.set_value( iter, 0, aNewlabel )
		key = str( self.theSysTreeStore.get_path( iter ) )
		self.theSysTreeStore.set_data( key, aSystemFullID )
		    
		aSystemPath = createSystemPathFromFullID( aSystemFullID )
		aSystemList = self.theSession.getEntityList( 'System',\
							     aSystemPath )
		aSystemListLength = len( aSystemList )

		if  aSystemListLength == 0:
			return
		
		for aSystemID in aSystemList:
			aNewSystemFullID = ( SYSTEM, aSystemPath, aSystemID )
			self.constructTree( iter, aNewSystemFullID )

			aPath = self.theSysTreeStore.get_path( iter )
			if aSystemListLength < 6 and len( aPath ) < 6:
				self.systemTree.expand_row( aPath, gtk.TRUE )
				

	def updateSystemSelection( self, obj=None ):
		self.theSelectedSystemIter = self.systemTree.get_selection().get_selected()[1]
		self.updateLists()

		if type(obj) == gtk.TreeView:

			sysTreeStore = self.theSysTreeStore
			selectedSystemIter = self.theSelectedSystemIter
			key=str( sysTreeStore.get_path( selectedSystemIter ) )
			systemFullID = sysTreeStore.get_data( key )
			systemPath = createSystemPathFromFullID( systemFullID )
			entityList= self.theSession.getEntityList( 'System',\
								   systemPath )

			aFullPN =  convertFullIDToFullPN( systemFullID ) 
			fullPN = convertFullIDToFullPN( systemFullID )
			self.thePropertyWindow.setRawFullPNList( [ fullPN ] )

	def getSelectedSystemList( self ):

		# get system ID from selected items of system tree,
		# and get entity list from session
		selectedSystemIter = self.theSelectedSystemIter
		key = str( self.theSysTreeStore.get_path( selectedSystemIter ) )
		systemFullID = self.theSysTreeStore.get_data( key )
		systemPath = createSystemPathFromFullID( systemFullID )

		return ( systemPath, )


	def updateLists( self ):
		if self.theSelectedSystemIter == None:
			return None

		selectedSystemList = self.getSelectedSystemList()

		# Variable list
		store = self.variableTree.get_model()
		self.updateEntityList( 'Variable', store,\
				       self.variableTree.get_columns(),\
				       selectedSystemList )

		# Process list
		store = self.processTree.get_model()
		self.updateEntityList( 'Process', store,
				       self.processTree.get_columns(),\
				       selectedSystemList )

		self.updateListLabels()


	def updateListLabels( self ):

		self.updateListLabel( 'Variable', self['variable_label'],\
				      self.variableTree )
		self.updateListLabel( 'Process', self['process_label'],\
				      self.processTree )

	def updateListLabel( self, type, label, view ):

		shownCount    = len( view.get_model() )
		selectedCount = view.get_selection().count_selected_rows()
		labelText = type + ' (' + str( selectedCount ) + '/' + \
			    str( shownCount ) + ')' 
		label.set_text( labelText )



	def updateEntityList( self, entityType, entityStore, columnList,\
			      systemList ):
		
		# get the entity list in the selected system(s)

		systemPath = systemList[0]

		if len( systemList ) == 1:
			shownPath = '.'
		else:
			shownPath = systemPath

		entityList = self.theSession.getEntityList( entityType,\
							    systemPath )

		# clear the store
		entityStore.clear()

		
		# re-create the list
		for ID in entityList:

			# temporary hack for the entity searching.
			# this can be like this in python 2.3 or above
			# if not self.searchString in ID:
			if ID.find( self.searchString ) < 0:
				continue

			iter = entityStore.append()

			typeID = ENTITYTYPE_DICT[ entityType ]
			fullID = ( typeID, systemPath, ID )
			fullIDString = createFullIDString( fullID )
			stub = self.theSession.createEntityStub( fullIDString )
			entityStore.set_data( ID, fullID )
			
			for i in range( len( columnList ) ):
				title = columnList[i].get_title()

				if title == 'ID':
					value = ID
				elif title == 'Classname':
					value = stub.getClassname()
				elif title == 'Path':
					value = shownPath
				else:  # usual entity property
					value = stub[ title ]

				entityStore.set_value( iter, i, str( value ) )




	def selectProcess( self, selection ):

		self.theLastSelectWindow = "Process"

		# clear fullPN list
		self.theSelectedFullPNList = []

		# get selected items
		selection.selected_foreach(self.process_select_func)
		if len(self.theSelectedFullPNList)>0:
			self.thePropertyWindow.setRawFullPNList( [self.theSelectedFullPNList[0]] )

		# clear selection of variable list
		self.variableTree.get_selection().unselect_all()

		self.updateListLabels()


	def selectVariable( self, selection ):

		self.theLastSelectWindow = "Variable"

		# clear fullPN list
		self.theSelectedFullPNList = []

		# get selected items
		selection.selected_foreach(self.variable_select_func)

		if len(self.theSelectedFullPNList)>0:
			self.thePropertyWindow.setRawFullPNList( [self.theSelectedFullPNList[0]] )

		# clear selection of process list
		self.processTree.get_selection().unselect_all()

		self.updateListLabels()

	def checkCreateLoggerButton(self):
		isSensitive = gtk.FALSE
		loggerList = self.theSession.getLoggerList()
		rawList = self.__getSelectedRawFullPNList()
		if rawList != None:
			for aFullPN in rawList:
				if aFullPN[3] == '':
					aFullPN = self.thePropertyWindow.supplementFullPN( aFullPN )
				aFullPNString = createFullPNString( aFullPN )
				aValue = self.theSession.theSimulator.getEntityProperty( aFullPNString )
				if not operator.isNumberType( aValue ):
					continue
				if aFullPNString not in loggerList:
					isSensitive = gtk.TRUE
					break
		self['logger_button'].set_sensitive( isSensitive )


	def variable_select_func(self,tree,path,iter):
		'''function for variable list selection

		Return None
		'''

		key=self.variableTree.get_model().get_value(iter,0)
		entityFullID = self.variableTree.get_model().get_data( key )
		entityFullPN = entityFullID + ( DEFAULT_VARIABLE_PROPERTY, )
		self.theSelectedFullPNList.append( entityFullPN )


	def process_select_func(self,tree,path,iter):
		'''function for process list selection

		Return None
		'''

		key=self.processTree.get_model().get_value(iter,0)
		entityFullID = self.processTree.get_model().get_data( key )
		entityFullPN = entityFullID + ( DEFAULT_PROCESS_PROPERTY, )
		self.theSelectedFullPNList.append( entityFullPN )



	def createPluginWindow( self, *obj ) :
		"""creates new PluginWindow instance(s)
		*obj   --  gtk.MenuItem on popupMenu or gtk.Button
		Returns None
		"""

		self.thePropertyWindow.clearStatusBar()

		if len(obj) == 0:
			return None

		aPluginWindowType = DEFAULT_WINDOW
		aSetFlag = FALSE

		# When this method is called by popup menu
		if type( obj[0] ) == gtk.MenuItem:
			aPluginWindowType = obj[0].get_name()

		# When this method is called by 'CreateWindow' button
		elif type( obj[0] ) == gtk.Button:
			aPluginWindowType = self['plugin_optionmenu'].get_children()[0].get()

		else:
			raise TypeErrir("%s must be gtk.MenuItem or gtk.Button" %str(type(obj[0])))

		aSelectedRawFullPNList = self.__getSelectedRawFullPNList()

		# When no FullPN is selected, displays error message.
		if aSelectedRawFullPNList  == None or len( aSelectedRawFullPNList ) == 0:

			aMessage = 'No entity is selected.'
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			self.thePropertyWindow.showMessageOnStatusBar(aMessage)
			return FALSE

		self.thePropertyWindow.setRawFullPNList( aSelectedRawFullPNList )
		self.thePluginManager.createInstance( aPluginWindowType, self.thePropertyWindow.theFullPNList() )



	def appendData( self, *obj ):
		"""appends RawFullPN to PluginWindow instance
		Returns TRUE(when appened) / FALSE(when not appened)
		"""

		# clear status bar
		self.thePropertyWindow.clearStatusBar()

		if len(obj) == 0:
			return None

		# Only when at least one menu is selected.

		# ----------------------------------------------------
		# When this method is called by popup menu
		# ----------------------------------------------------
		if type( obj[0] ) == gtk.MenuItem:
			aSetFlag = TRUE
			aPluginWindowTitle = obj[0].get_name()

			for anInstance in self.thePluginManager.theInstanceList:
				if anInstance.getTitle() == aPluginWindowTitle:

					try:
						anInstance.appendRawFullPNList( self.__getSelectedRawFullPNList() )
					except TypeError:
						anErrorFlag = TRUE
						aMessage = "Can't append data to %s" %str(anInstance.getTitle())
						self.thePropertyWindow.showMessageOnStatusBar(aMessage)
					else:
						aMessage = "Selected Data are added to %s" %aPluginWindowTitle
						self.thePropertyWindow.showMessageOnStatusBar(aMessage)
					break
					
			return TRUE

		# ----------------------------------------------------
		# When this method is called by PluginInstanceWindow
		# ----------------------------------------------------
		elif type( obj[0] ) == gtk.Button:

			self.theSelectedPluginInstanceList = []
			selection=self.thePluginInstanceSelection['plugin_tree'].get_selection()
			selection.selected_foreach(self.thePluginInstanceSelection.plugin_select_func)

			# When no FullPN is selected, displays error message.
			if self.__getSelectedRawFullPNList() == None or len( self.__getSelectedRawFullPNList() ) == 0:

				aMessage = 'No entity is selected.'
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				self.thePropertyWindow.showMessageOnStatusBar(aMessage)
				return FALSE

			# When no plugin instance is selected, displays error message.
			if len(self.theSelectedPluginInstanceList) == 0:

				aMessage = 'No Plugin Instance is selected.'
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				self.thePropertyWindow.showMessageOnStatusBar(aMessage)
				return FALSE

			# buffer of appended instance's title
			anAppendedTitle = []

			anErrorFlag = FALSE

			# appneds data
			for aPluginWindowTitle in self.theSelectedPluginInstanceList:
				for anInstance in self.thePluginManager.theInstanceList:
					if anInstance.getTitle() == aPluginWindowTitle:
						try:
							anInstance.appendRawFullPNList( self.__getSelectedRawFullPNList() )
						except TypeError:
							anErrorFlag = TRUE
							aMessage = "Can't append data to %s" %str(anInstance.getTitle())
							self.thePropertyWindow.showMessageOnStatusBar(aMessage)
						else:
							anAppendedTitle.append( anInstance.getTitle() )
						break

			# When at least one instance is appended
			if len(anAppendedTitle) > 0 and anErrorFlag == FALSE:
				# displays message
				aMessage = "Selected Data are added to %s" %str(anAppendedTitle)
				self.theSession.message(aMessage)
				self.thePropertyWindow.showMessageOnStatusBar(aMessage)

				# closes PluginInstanceSelectionWindow
				#self.__closePluginInstanceSelectionWindow()
				self.closePluginInstanceSelectionWindow()
				return TRUE

			# When no instance is appended
			else:

				return NONE



	def __getSelectedRawFullPNList( self ):
		"""
		Return a list of selected FullPNs
		"""

		self.theSelectedFullPNList = []

		if ( self.theLastSelectWindow == "None" ):
			return None

		if ( self.theLastSelectWindow == "Variable" ):
		
			selection=self.variableTree.get_selection()
			selection.selected_foreach(self.variable_select_func)

		if ( self.theLastSelectWindow == "Process" ):
			
			selection=self.processTree.get_selection()
			selection.selected_foreach(self.process_select_func)

		if len(self.theSelectedFullPNList) == 0:
			aSelectedSystemIter = self.systemTree.get_selection().get_selected()[1]
			if aSelectedSystemIter != None:
				key=str(self.theSysTreeStore.get_path(aSelectedSystemIter))
				aSystemFullID = self.theSysTreeStore.get_data( key )
				self.theSelectedFullPNList = [(aSystemFullID[0],aSystemFullID[1],aSystemFullID[2],'')]


		# If no property is selected on PropertyWindow, 
		# create plugin Window with default property (aValue) 
		if len( str(self.thePropertyWindow.getSelectedFullPN()) ) == 0:
			return self.theSelectedFullPNList

		# If a property is selected on PropertyWindow, 
		# create plugin Window with selected property
		else:
			return [self.thePropertyWindow.getSelectedFullPN()]




	def addToBoard( self, *arg ):
		"""add plugin window to board
		"""

		self.thePropertyWindow.clearStatusBar()

		if len(arg) == 0:
			return None

		aPluginWindowType = DEFAULT_WINDOW
		aSetFlag = FALSE

		# When this method is called by popup menu
		if type( arg[0] ) == gtk.MenuItem:
			aPluginWindowType = arg[0].get_name()

		# When this method is called by 'CreateWindow' button
		elif type( arg[0] ) == gtk.Button:
			aPluginWindowType = self['plugin_optionmenu'].get_children()[0].get()

		else:
			raise TypeError("%s must be gtk.MenuItem or gtk.Button" %str(type(arg[0])))

		aSelectedRawFullPNList = self.__getSelectedRawFullPNList()

		# When no FullPN is selected, displays error message.
		if aSelectedRawFullPNList  == None or len( aSelectedRawFullPNList ) == 0:

			aMessage = 'No entity is selected.'
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			self.thePropertyWindow.showMessageOnStatusBar(aMessage)
			return FALSE

		self.theSession.getWindow('BoardWindow').addPluginWindows( aPluginWindowType, \
		self.__getSelectedRawFullPNList() )



	def createLogger( self, *arg ):
		"""creates Logger about all FullPN selected on EntityTreeView
		Returns None
		"""

		# clear status bar
		self.thePropertyWindow.clearStatusBar()

		# gets selected RawFullPNList
		aSelectedRawFullPNList = self.__getSelectedRawFullPNList()


		# When no entity is selected, displays confirm window.
		if len(aSelectedRawFullPNList) == 0:

			# print message to message 
			aMessage = 'No Entity is selected.'
			self.thePropertyWindow.showMessageOnStatusBar(aMessage)
			aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
			return None

		# creates Logger using PropertyWindow
		self.thePropertyWindow.setRawFullPNList( aSelectedRawFullPNList )
		self.thePropertyWindow.createLogger()

		# display message on status bar
		if len(aSelectedRawFullPNList) == 1:
			aMessage = 'Logger was created.'
		else:
			aMessage = 'Loggers were created.'
		self.thePropertyWindow.showMessageOnStatusBar(aMessage)
		#self.checkCreateLoggerButton()



	def searchEntity( self ):
		"""search Entities
		Returns None
		"""
		if self.theSelectedSystemIter == None:
			return

		self.searchString = self['search_entry'].get_text()

		self.updateLists()


	def pushSearchButton( self, *arg ):
		self.searchEntity()

	def keypressOnSearchEntry( self, *arg ):

		if( arg[1].keyval == 65293 ):

			self.searchEntity()
