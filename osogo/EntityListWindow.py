#!/usr/bin/env python

from OsogoWindow import *

import gtk
from ecell.ecssupport import *
import gobject
import MainWindow

import string
import copy

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

		# call constructor of super class
		OsogoWindow.__init__( self, aMainWindow )
		OsogoWindow.openWindow( self )


		# create popupmenu
		self.thePopupMenu = PopupMenu( aMainWindow.thePluginManager,\
					       self )
		# add handers
		self.addHandlers( { 'show_button_clicked' :\
				    self.openNewPluginWindow,\
		                    'system_tree_cursor_changed' :\
				    self.updateSystemSelection,\
		                    'entity_list_cursor_changed' :\
				    self.selectEntity,\
		                    'system_tree_button_press_event'    :\
				    self.popupMenu,\
		                    'entity_list_button_press_event'    :\
				    self.popupMenu\
				    } )


		self.thePaletteWindow = aMainWindow.thePaletteWindow

		# initialize widgets
		self.theSystemTree = self.getWidget( 'system_tree' )
		self.theEntityList = self.getWidget( 'entity_list' )
		self.theTypeOptionMenu = self.getWidget( 'type_optionmenu' )
		self.theStatusBar = self.getWidget( 'statusbar' )

		self.theTypeMenu = self.theTypeOptionMenu.get_menu()
		self.theTypeMenu.connect( 'selection-done',\
					  self.updateEntityList )

		aTypeMenuItemMap = self.theTypeMenu.get_children()
		aTypeMenuItemMap[0].set_data( 'LABEL', 'Variable' )
		aTypeMenuItemMap[1].set_data( 'LABEL', 'Process' )
		aTypeMenuItemMap[2].set_data( 'LABEL', 'All' )

		self.theSystemTree.show()
		self.theEntityList.show()

		self.displayed_depth=-1
		
		self.theSysTreeStore=gtk.TreeStore( gobject.TYPE_STRING )
		self.theSystemTree.set_model(self.theSysTreeStore)
				    
		self.theEntityListStore=gtk.ListStore( gobject.TYPE_STRING )
		self.theEntityList.get_selection().set_mode( gtk.SELECTION_MULTIPLE )
		self.theEntityList.set_model( self.theEntityListStore )
		
		column=gtk.TreeViewColumn( 'System Tree',
					   gtk.CellRendererText(),
					   text=0 )
		column.set_visible( gtk.TRUE )
		self.theSystemTree.append_column(column)

		column = gtk.TreeViewColumn( 'Entity List',
					     gtk.CellRendererText(),
					     text=0 )
		self.theEntityList.append_column(column)

		aPManager = self.theMainWindow.thePluginManager
		self.thePropertyWindow = aPManager.createInstance( 'PropertyWindow', [(4, '', '/', '')], 'top_vbox' ) 
		aPropertyWindowTopVBox = self.thePropertyWindow['top_vbox']
		self['property_frame'].add( aPropertyWindowTopVBox )
		self.thePropertyWindow['property_clist'].connect( 'select_cursor_row', self.selectPropertyName )

		self.theSysTreeStore.clear()
		aRootSystemFullID = createFullID( 'System::/' )
		self.constructTree( None, aRootSystemFullID )
		self.update()


	def update( self ):
		self.theEntityListStore.clear()

		self.updateSystemSelection()
		self.thePropertyWindow.update()

            
	def constructTree( self, aParentTree, aSystemFullID ):
		newlabel = aSystemFullID[ID] 
	 
		iter  = self.theSysTreeStore.append( aParentTree )
#		depth = self.theSysTreeStore.iter_depth ( iter )

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

			if aSystemListLength <= 5:
				aPath = self.theSysTreeStore.get_path( iter )
				self.theSystemTree.expand_row( aPath,
							       gtk.FALSE )


	def updateSystemSelection( self, obj=None ):
		aSelectedSystemIter = self.theSystemTree.get_selection().get_selected()[1]
		if aSelectedSystemIter == None:
			return

		self.updateEntityList( aSelectedSystemIter )

#		self.selectSystem( aSelectedSystemIter )


	def updateEntityList( self, aSelectedSystemIter ):

		aSelectedTypeMenuItem = self.theTypeMenu.get_active()
		aEntityTypeString = aSelectedTypeMenuItem.get_data( 'LABEL' )
		key=str(self.theSysTreeStore.get_path(aSelectedSystemIter))
    		aSystemFullID = self.theSysTreeStore.get_data( key )
		self.theEntityListStore.clear()
		if aEntityTypeString == 'All':
			self.listEntity( 'Variable', aSystemFullID )
			self.listEntity( 'Process', aSystemFullID )
		else:
			self.listEntity( aEntityTypeString, aSystemFullID )

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
			print self.theEntityListStore.get_value(iter,0)

	def selectEntity( self, aEntityList ):
		self.theSelectedFullPNList = []

		aSelection = aEntityList.get_selection()
		aSelection.selected_foreach(self.entity_select_func)
		if len(self.theSelectedFullPNList)>0:
		    self.thePropertyWindow.theRawFullPNList = self.theSelectedFullPNList
		    self.thePropertyWindow.setFullPNList()
		    self.updateStatusBar()
		
	def entity_select_func(self,tree,path,iter):
		key=self.theEntityListStore.get_value(iter,0)
		aEntityFullPN = self.theEntityListStore.get_data( key )
		self.theSelectedFullPNList.append( aEntityFullPN )
	
	def selectPropertyName( self, aCList, row, column, event_obj ):

		self.theSelectedFullPNList = []
		for aRowNumber in aCList.selection:
			aPropertyName =  aCList.get_text( aRowNumber, 0 )
			aFullID = self.thePropertyWindow.theFullID()
			aFullPN = convertFullIDToFullPN( aFullID, aPropertyName )
			self.theSelectedFullPNList.append( aFullPN )
		self.updateStatusBar()
        
	def selectSystem( self, iter ):
		key = str( self.theSysTreeStore.get_path( iter ) )
		aFullID = self.theSysTreeStore.get_data( key )
		aFullPN = convertFullIDToFullPN( aFullID )
		self.theSelectedFullPNList = [ aFullPN ]
		self.thePropertyWindow.theRawFullPNList = self.theSelectedFullPNList
		self.thePropertyWindow.setFullPNList()
		self.updateStatusBar()

	def updateStatusBar( self ):
		aStatusString = 'Selected: '
		for aFullPN in self.theSelectedFullPNList:
			aStatusString += createFullPNString( aFullPN )
			aStatusString += ', '
		self.theStatusBar.push( 1, aStatusString )

	def openNewPluginWindow( self, obj ) :

		try:
			aPluginName = self.thePaletteWindow.getSelectedPluginName()
			aPluginManager = self.theMainWindow.thePluginManager
			aPluginManager.createInstance( aPluginName, self.theSelectedFullPNList )

		except:
			self.theMainWindow.printMessage('Error: couldn\'t create plugin window')
		else:
			self.theMainWindow.printMessage('create new plugin window ')
			
	# end of openNewPluginWindow

	

	# ---------------------------------------------------------------
	# openNewPluginWindowByPopupMenu
	#   - gets FullPN via PropertyWindow 
	#   - creates PluginWindow
	#
	# anObject          : selected menu item
	#
	# return -> None
	# ---------------------------------------------------------------

	def openNewPluginWindowByPopupMenu(self, anObject ):

		#try:

			aPluginManager = self.theMainWindow.thePluginManager
			aPluginManager.createInstance( anObject.get_name(), self.theSelectedFullPNList)

		#except:
			#self.theMainWindow.printMessage('Error: couldn\'t create plugin window')
		#else:
			self.theMainWindow.printMessage('create new plugin window ')


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

		self.thePropertyWindow.theRawFullPNList = self.theSelectedFullPNList
		self.thePropertyWindow.createLogger()

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
			self.theMenuItem[aPluginMap].connect('activate', self.theParent.openNewPluginWindowByPopupMenu )
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
		self.theHeight = (aMenuSize+1)*21 + 3
		self.set_usize( self.theWidth, self.theHeight )

	# end of __init__

	def popup(self, pms, pmi, func, button, time):
		gtk.Menu.popup(self, pms, pmi, func, button, time)
		self.show_all()
	
if __name__ == "__main__":

        def mainQuit( obj, data ):
                gtk.mainquit()

        def mainLoop():
                gtk.mainloop()

        def main():
                aWindow = EntityListWindow( 'EntityListWindow.glade' )
                aWindow.addHandler( 'gtk_main_quit', mainQuit )
                mainLoop()

        def ecstest():
                aWindow = EntityListWindow( 'EntityListWindow.glade' )
                aWindow.addHandler( 'gtk_main_quit', mainQuit )
                mainLoop()

        main()

if __name__ == "__ecstest__":
        ecstest()
