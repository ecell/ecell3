#!/usr/bin/env python

from OsogoWindow import *

import gtk
from ecell.ecssupport import *
import gobject
import MainWindow

import string
import copy

# This parameter should be set by setting file.
DEFAULT_WINDOW = 'TracerWindow'

# ---------------------------------------------------------------
# EntityListWindow -> OsogoWindow
# ---------------------------------------------------------------
class EntityListWindow(OsogoWindow):

	# ---------------------------------------------------------------
	# Constructor
	#
	# aMainWindow : reference to MainWindow
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aMainWindow ):

		# initialize parameters
		self.theSelectedFullPNList = []
		self.thePluginManager = aMainWindow.thePluginManager
		self.theSession = aMainWindow.theSession

		# call constructor of super class
		OsogoWindow.__init__( self, aMainWindow )
		OsogoWindow.openWindow( self )

		# create popupmenu
		self.thePopupMenu = PopupMenu( aMainWindow.thePluginManager, self )
		# add handers
		self.addHandlers( { 
		                    # plugin button
		                    #'plugin_button_clicked'      : self.openNewPluginWindow,\
		                    # system tree
		                    'on_system_tree_cursor_changed' : self.updateSystemSelection,\
		                    'on_system_tree_button_pressed' : self.popupMenu,\
		                    # entity list
		                    'on_entity_tree_cursor_changed' : self.selectEntity,\
		                    'on_entity_tree_button_pressed' : self.popupMenu,\
		                    #
		                    'on_entity_optionmenu_changed' : self.updateSystemSelection,\
		                    # create button
		                    'on_create_button_clicked'     : self.createPluginWindow,\
				    } )

		#self.thePaletteWindow = aMainWindow.thePaletteWindow

		# initialize widgets
		#self.theTypeMenu = self.theTypeOptionMenu.get_menu()
		#self.theTypeMenu.connect( 'selection-done',\
		#			  self.updateSystemSelection )

		#aTypeMenuItemMap = self.theTypeMenu.get_children()
		#aTypeMenuItemMap[0].set_data( 'LABEL', 'Variable' )
		#aTypeMenuItemMap[1].set_data( 'LABEL', 'Process' )
		#aTypeMenuItemMap[2].set_data( 'LABEL', 'All' )

		#self.theSystemTree.show()
		#self.theEntityList.show()

		#self.displayed_depth=-1
		
		self.theSysTreeStore=gtk.TreeStore( gobject.TYPE_STRING )
		self['system_tree'].set_model(self.theSysTreeStore)
				    
		#self.theEntityListStore=gtk.ListStore( gobject.TYPE_STRING )
		#self.theEntityList.get_selection().set_mode( gtk.SELECTION_MULTIPLE )
		#self.theEntityList.set_model( self.theEntityListStore )
		
		# ------------------------------------------
		# setup system tree
		# ------------------------------------------
		column=gtk.TreeViewColumn( 'System Tree',
					   gtk.CellRendererText(),
					   text=0 )
		column.set_visible( gtk.TRUE )
		self['system_tree'].append_column(column)


		# ------------------------------------------
		# setup entity tree
		# ------------------------------------------
		column = gtk.TreeViewColumn( 'Entity List',
					     gtk.CellRendererText(),
					     text=0 )
		self['entity_tree'].append_column(column)

		self.theEntityListStore=gtk.ListStore( gobject.TYPE_STRING )
		self['entity_tree'].get_selection().set_mode( gtk.SELECTION_MULTIPLE )
		self['entity_tree'].set_model( self.theEntityListStore )

		# ------------------------------------------
		# setup plugin_menu
		# ------------------------------------------
		aBuffer = []
		aDefaultFlag = FALSE
		aPluginWindowNameList = []
		for aPluginWindowName in self.thePluginManager.thePluginMap.keys():
			if aPluginWindowName == DEFAULT_WINDOW:
				aDefaultFlag = TRUE
			else:
				aBuffer.append(aPluginWindowName)

		if aDefaultFlag == TRUE:
			aPluginWindowNameList.append( DEFAULT_WINDOW )

		aPluginWindowNameList += aBuffer 

		aMenu = gtk.Menu()
		#for aPluginWindowName in self.thePluginManager.thePluginMap.keys():
		for aPluginWindowName in aPluginWindowNameList:
			aMenuItem = gtk.MenuItem(aPluginWindowName)
			aMenuItem.show()
			aMenu.append( aMenuItem )
		self['plugin_menu'].set_menu(aMenu)
		self['plugin_menu'].show()

		# set default menu

		# ------------------------------------------
		# setup PropertyWindow
		# ------------------------------------------
		self.thePropertyWindow = self.thePluginManager.createInstance( \
		                         'PropertyWindow', [(SYSTEM, '', '/', '')], 'top_vbox' ) 
		self.thePropertyWindow.setStatusBar( self['statusbar'] )

		aPropertyWindowTopVBox = self.thePropertyWindow['top_vbox']
		self['property_area'].add( aPropertyWindowTopVBox )
		self.thePropertyWindow['property_clist'].connect( 'select_cursor_row', self.selectPropertyName )
		self.theSysTreeStore.clear()

		# create route ID
		aRootSystemFullID = createFullID( 'System::/' )
		self.constructTree( None, aRootSystemFullID )
		self.update()

		aSelection = self['entity_tree'].get_selection()

		self.theSelectedEntityList = []

	# ---------------------------------------------------------------
	# update
	#
	# return -> None
	# ---------------------------------------------------------------
	def update( self ):
		OsogoWindow.update(self)
		self.thePropertyWindow.update()

	# end of update


	# ---------------------------------------------------------------
	# constructTree 
	#
	# return -> None
	# ---------------------------------------------------------------
	def constructTree( self, aParentTree, aSystemFullID ):

		newlabel = aSystemFullID[ID] 
		iter  = self.theSysTreeStore.append( aParentTree )

		self.theSysTreeStore.set_value( iter, 0, newlabel )
		key = str( self.theSysTreeStore.get_path( iter ) )
		self.theSysTreeStore.set_data( key, aSystemFullID )
		    
		aSystemListFullPN = convertFullIDToFullPN( aSystemFullID, 'SystemList' ) 
		aSystemList = self.theMainWindow.theSession.theSimulator.getEntityProperty( createFullPNString( aSystemListFullPN ) )
		aSystemListLength = len( aSystemList )

		if  aSystemListLength != 0:

			for aSystemID in aSystemList:
				aSystemPath = createSystemPathFromFullID( aSystemFullID )
				aNewSystemFullID = ( SYSTEM, aSystemPath, aSystemID )
				self.constructTree( iter, aNewSystemFullID )


	def updateSystemSelection( self, obj=None ):
		aSelectedSystemIter = self['system_tree'].get_selection().get_selected()[1]
		if aSelectedSystemIter == None:
			return None

		self.updateEntityList( aSelectedSystemIter )

		if type(obj) == gtk.TreeView:
			anEntityTypeString = self['entity_optionmenu'].get_children()[0].get()
			key=str(self.theSysTreeStore.get_path(aSelectedSystemIter))
			aSystemFullID = self.theSysTreeStore.get_data( key )
			aFullPN =  convertFullIDToFullPN(aSystemFullID) 
			self.thePropertyWindow.setRawFullPNList( [convertFullIDToFullPN(aSystemFullID)] )

	def updateEntityList( self, aSelectedSystemIter ):

		# ---------------------------------
		# gets string of selected item
		# ---------------------------------
		# GtkContainer.children() is deprecated
		anEntityTypeString = self['entity_optionmenu'].get_children()[0].get()
		key=str(self.theSysTreeStore.get_path(aSelectedSystemIter))
		aSystemFullID = self.theSysTreeStore.get_data( key )
		self.theEntityListStore.clear()

		if anEntityTypeString == 'All Entities':
			self.listEntity( 'Variable', aSystemFullID )
			self.listEntity( 'Process', aSystemFullID )
		else:
			self.listEntity( anEntityTypeString, aSystemFullID )

	# create list of Entity tree
	def listEntity( self, aEntityTypeString, aSystemFullID ):

		aListPN = aEntityTypeString + 'List'
		aListFullPN = convertFullIDToFullPN( aSystemFullID, aListPN )
		aEntityList = self.theMainWindow.theSession.theSimulator.getEntityProperty( createFullPNString( aListFullPN ) )

		if aEntityTypeString == 'Variable':
			aEntityType = VARIABLE
		elif aEntityTypeString == 'Process':
			aEntityType = PROCESS

		for anEntityID in aEntityList:
			iter = self.theEntityListStore.append()
			self.theEntityListStore.set_value(iter,0,anEntityID)

			aSystemPath = createSystemPathFromFullID( aSystemFullID )
			aEntityFullPN = ( aEntityType, aSystemPath, anEntityID, '' )
			self.theEntityListStore.set_data( anEntityID, aEntityFullPN )

	def selectEntity( self, aEntityList ):
		self.theSelectedFullPNList = []
		aSelection = aEntityList.get_selection()
		aSelection.selected_foreach(self.entity_select_func)
		if len(self.theSelectedFullPNList)>0:
			self.thePropertyWindow.setRawFullPNList( [self.theSelectedFullPNList[0]] )

	def entity_select_func(self,tree,path,iter):
		key=self.theEntityListStore.get_value(iter,0)
		aEntityFullPN = self.theEntityListStore.get_data( key )
		self.theSelectedFullPNList.append( aEntityFullPN )

	# end of entity_select_func


	def createPluginWindow( self, *obj ) :

		aPluginWindowType = DEFAULT_WINDOW
		aSetFlag = FALSE

		if len(obj) >= 1:
			# This method is called by popup menu
			if type( obj[0] ) == gtk.MenuItem:
				aSetFlag = TRUE
				aPluginWindowType = obj[0].get_name()
			else:
				pass

		# This method is called by 'Create Window' button
		if aSetFlag == FALSE:
			aPluginWindowType = self['plugin_menu'].get_children()[0].get()

		self.theSelectedFullPNList = []

		selection=self['entity_tree'].get_selection()
		selection.selected_foreach(self.entity_select_func)

		# If no property is selected on PropertyWindow, create plugin Window
		# with default property (aValue) 
		if len( str(self.thePropertyWindow.getSelectedFullPN()) ) == 0:  # if(1)
			self.thePluginManager.createInstance( aPluginWindowType, self.theSelectedFullPNList )

		# If a property is selected on PropertyWindow, create plugin Window
		# with selected property
		else:  # if(1)

			aSpecifiedProperty = self.thePropertyWindow.getSelectedFullPN()[PROPERTY]
			# buffer list for FullPN that doen not have specified property
			aNoPropertyFullIDList = []
			aSelectedFullPNListWithSpecified = []

			for aSelectedFullPN in self.theSelectedFullPNList:
				aFullID = convertFullPNToFullID(aSelectedFullPN) 
				aFullIDString = createFullIDString( aFullID )
				aEntityStub = EntityStub(self.theSession.theSimulator,aFullIDString)
				aPropertyExistsFlag = FALSE
				for aProperty in aEntityStub.getPropertyList():
					if aProperty == aSpecifiedProperty:
						aFullPN = convertFullIDToFullPN( aFullID, aSpecifiedProperty )
						aSelectedFullPNListWithSpecified.append( aFullPN )
						aPropertyExistsFlag = TRUE
						break
				if aPropertyExistsFlag == FALSE:
					aNoPropertyFullIDList.append( aFullIDString )

			# When some selected Entity does not have specified property,
			# shows confirmWindow and does not create plugin window.
			if len(aNoPropertyFullIDList) != 0:
				aTitile = ' Error !'

				# creates message
				aMessage = ''
				# one entity
				if len(aNoPropertyFullIDList) == 1:
					aMessage += ' The following Entity does not have %s \n' %aSpecifiedProperty
				# entities
				else:
					aMessage += ' The following Entities do not have %s \n' %aSpecifiedProperty

				for aNoPropertyFullIDString in aNoPropertyFullIDList:
					aMessage += aNoPropertyFullIDString + '\n'

				# print message to message 
				self.theMainWindow.printMessage(aMessage)
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				return None


			# creates plugin window
			self.thePluginManager.createInstance( aPluginWindowType, aSelectedFullPNListWithSpecified )

		# end of if(1)
			
	# end of openNewPluginWindow

	def selectPropertyName( self, aCList, row, column, event_obj ):

		self.theSelectedFullPNList = []
		for aRowNumber in aCList.selection:
			aPropertyName =  aCList.get_text( aRowNumber, 0 )
			aFullID = self.thePropertyWindow.theFullID()
			aFullPN = convertFullIDToFullPN( aFullID, aPropertyName )
			self.theSelectedFullPNList.append( aFullPN )
			self.updateStatusBar()

	# end of selectPropertyName

	# ---------------------------------------------------------------
	# popupMenu
	#   - show popup menu
	#
	# aWidget         : widget
	# anEvent          : an event
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def popupMenu( self, aWidget, anEvent ):

		if anEvent.button == 3:
			self.thePopupMenu.popup( None, None, None, 1, 0 )

	# end of poppuMenu

	# ---------------------------------------------------------------
	# createLogger
	#   - create Logger
	#
	# *objects : dammy objects
	#
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def createLogger( self, *objects ):

		# If no property is selected on PropertyWindow, create plugin Window
		# with default property (aValue) 
		if len( str(self.thePropertyWindow.getSelectedFullPN()) ) == 0:  # if(1)
			self.thePropertyWindow.theRawFullPNList = self.theSelectedFullPNList
			self.thePropertyWindow.createLogger()
		# If a property is selected on PropertyWindow, create plugin Window
		# with selected property
		else:  # if(1)

			aSpecifiedProperty = self.thePropertyWindow.getSelectedFullPN()[PROPERTY]
			# buffer list for FullPN that doen not have specified property
			aNoPropertyFullIDList = []
			aSelectedFullPNListWithSpecified = []

			for aSelectedFullPN in self.theSelectedFullPNList:
				aFullID = convertFullPNToFullID(aSelectedFullPN) 
				aFullIDString = createFullIDString( aFullID )
				aEntityStub = EntityStub(self.theSession.theSimulator,aFullIDString)
				aPropertyExistsFlag = FALSE
				for aProperty in aEntityStub.getPropertyList():
					if aProperty == aSpecifiedProperty:
						aFullPN = convertFullIDToFullPN( aFullID, aSpecifiedProperty )
						aSelectedFullPNListWithSpecified.append( aFullPN )
						aPropertyExistsFlag = TRUE
						break
				if aPropertyExistsFlag == FALSE:
					aNoPropertyFullIDList.append( aFullIDString )

			# When some selected Entity does not have specified property,
			# shows confirmWindow and does not create plugin window.
			if len(aNoPropertyFullIDList) != 0:
				aTitile = ' Error !'

				# creates message
				aMessage = ''
				# one entity
				if len(aNoPropertyFullIDList) == 1:
					aMessage += ' The following Entity does not have %s \n' %aSpecifiedProperty
				# entities
				else:
					aMessage += ' The following Entities do not have %s \n' %aSpecifiedProperty

				for aNoPropertyFullIDString in aNoPropertyFullIDList:
					aMessage += aNoPropertyFullIDString + '\n'

				# print message to message 
				self.theMainWindow.printMessage(aMessage)
				aDialog = ConfirmWindow(OK_MODE,aMessage,'Error!')
				return None


			# creates plugin window
			#self.thePluginManager.createInstance( aPluginWindowType, aSelectedFullPNListWithSpecified )
			self.thePropertyWindow.theRawFullPNList = aSelectedFullPNListWithSpecified
			self.thePropertyWindow.createLogger()

		# end of if(1)
			
	# end of openNewPluginWindow

	def selectPropertyName( self, aCList, row, column, event_obj ):

		self.theSelectedFullPNList = []
		for aRowNumber in aCList.selection:
			aPropertyName =  aCList.get_text( aRowNumber, 0 )
			aFullID = self.thePropertyWindow.theFullID()
			aFullPN = convertFullIDToFullPN( aFullID, aPropertyName )
			self.theSelectedFullPNList.append( aFullPN )

	# end of createLogger



# ---------------------------------------------------------------
# PopupMenu -> gtk.Menu
# ---------------------------------------------------------------
class PopupMenu( gtk.Menu ):

	# ---------------------------------------------------------------
	# Constructor
	#
	# aParent        : parent plugin window
	# return -> None
	# This method is throwable exception.
	# ---------------------------------------------------------------
	def __init__( self, aPluginManager, aParent ):

		# ------------------------------------------
		# calls the constructor of super class
		# ------------------------------------------
		gtk.Menu.__init__(self)
		self.theLoggerString = "add logger list"

		# ------------------------------------------
		# calls the constructor of super class
		# ------------------------------------------
		gtk.Menu.__init__(self)

		# ------------------------------------------
		# initializes the size of menu
		# ------------------------------------------
		aMaxStringLength = 0
		aMenuSize = 0

		# ------------------------------------------
		# sets arguments to instance valiables
		# ------------------------------------------
		self.theParent = aParent
		self.thePluginManager = aPluginManager

		# ------------------------------------------
		# initializes menu item
		# ------------------------------------------
		self.theMenuItem = {}

		# ------------------------------------------
		# adds all plugin window name to mete item
		# ------------------------------------------
		for aPluginMap in self.thePluginManager.thePluginMap.keys(): #(1)
			self.theMenuItem[aPluginMap]= gtk.MenuItem(aPluginMap)
			self.theMenuItem[aPluginMap].connect('activate', self.theParent.createPluginWindow )
			self.theMenuItem[aPluginMap].set_name(aPluginMap)
			self.append( self.theMenuItem[aPluginMap] )
			if aMaxStringLength < len(aPluginMap):
				aMaxStringLength = len(aPluginMap)
			aMenuSize += 1
		self.append( gtk.MenuItem() )

		# ------------------------------------------
		# adds creates logger
		# ------------------------------------------
		self.theMenuItem[self.theLoggerString]= gtk.MenuItem(self.theLoggerString)
		self.theMenuItem[self.theLoggerString].connect('activate', self.theParent.createLogger )
		self.theMenuItem[self.theLoggerString].set_name(self.theLoggerString)
		self.append( self.theMenuItem[self.theLoggerString] )

		# ------------------------------------------
		# caliculates size of menu and sets it to itself
		# ------------------------------------------
		self.theWidth = (aMaxStringLength+1)*8
		#self.theHeight = (aMenuSize+1)*21 + 3
		self.theHeight = (aMenuSize+1)*24 + 3
		#self.set_usize( self.theWidth, self.theHeight )
		self.set_size_request( self.theWidth, self.theHeight )

	# end of __init__

	def popup(self, pms, pmi, func, button, time):
		gtk.Menu.popup(self, pms, pmi, func, button, time)
		self.show_all()
	
