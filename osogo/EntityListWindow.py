#!/usr/bin/env python

from OsogoWindow import *

from gtk import *
from ecell.ecssupport import *

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

		# initializes parameters
		self.theSelectedFullPNList = []

		# calls constructor of super class
		OsogoWindow.__init__( self, aMainWindow )
		OsogoWindow.openWindow( self )


		# creates popupmenu
		self.thePopupMenu = PopupMenu( aMainWindow.thePluginManager, self )

		# adds handers
		self.addHandlers( { 'show_button_clicked'           : self.openNewPluginWindow,
		                    'system_tree_selection_changed' : self.updateEntityList,
		                    'entity_list_selection_changed'  : self.selectEntity,
		                    'tree_button_press_event'    : self.popupMenu,
		                    'list_button_press_event'    : self.popupMenu,
		                     } )


		#self.theMainWindow = aMainWindow
		self.thePaletteWindow = aMainWindow.thePaletteWindow

		# initialize widget
		self.theSystemTree = self.getWidget( 'system_tree' )
		self.theEntityList = self.getWidget( 'entity_list' )
		self.theTypeOptionMenu = self.getWidget( 'type_optionmenu' )
		self.theStatusBar = self.getWidget( 'statusbar' )

		self.theTypeMenu = self.theTypeOptionMenu.get_menu()
		self.theTypeMenu.connect( 'selection-done', self.updateEntityList)

		aTypeMenuItemMap = self.theTypeMenu.children()
		aTypeMenuItemMap[0].set_data( 'LABEL', 'Variable' )
		aTypeMenuItemMap[1].set_data( 'LABEL', 'Process' )
		aTypeMenuItemMap[2].set_data( 'LABEL', 'All' )

		self.theSystemTree.show()
		self.theEntityList.show()

		aPManager = self.theMainWindow.thePluginManager
		self.thePropertyWindow = aPManager.createInstance( 'PropertyWindow', [(4, '', '/', '')],'top_vbox' ) 
		aPropertyWindowTopVBox = self.thePropertyWindow['top_vbox']
		self['property_frame'].add( aPropertyWindowTopVBox )
		self.thePropertyWindow['property_clist'].connect( 'select_row', self.selectPropertyName )
		self.update()


	def update( self ):
		self.theSystemTree.clear_items(0, 999)
		self.theEntityList.clear_items(0, 999)
		aRootSystemFullID = createFullID( 'System::/' )
		self.constructTree( self.theSystemTree, aRootSystemFullID )
		self.updateEntityList()
		self.thePropertyWindow.update()

            
	def constructTree( self, aParentTree, aSystemFullID ):
		aLeaf = gtk.GtkTreeItem( label=aSystemFullID[ID] )
		aLeaf.set_data( 'FULLID', aSystemFullID )
		aLeaf.connect( 'select', self.selectSystem )
		aParentTree.append( aLeaf )
		aLeaf.show()

		aSystemListFullPN = convertFullIDToFullPN( aSystemFullID, 'SystemList' ) 
		aSystemList = self.theMainWindow.theSession.theSimulator.getEntityProperty( createFullPNString( aSystemListFullPN ) )
		if aSystemList != ():
			aTree = gtk.GtkTree()
			aLeaf.set_subtree( aTree )
			aLeaf.expand()

			for aSystemID in aSystemList:
				aSystemPath = createSystemPathFromFullID( aSystemFullID )
				aNewSystemFullID = ( SYSTEM, aSystemPath, aSystemID )
				self.constructTree( aTree, aNewSystemFullID )

	def updateEntityList( self, obj=None ):

		aSelectedSystemLeafMap = self.theSystemTree.get_selection()
		if aSelectedSystemLeafMap == [] :
			return
        
		aSelectedTypeMenuItem = self.theTypeMenu.get_active()
		#aPrimitiveTypeString = aSelectedTypeMenuItem.get_data( 'LABEL' )
		aEntityTypeString = aSelectedTypeMenuItem.get_data( 'LABEL' )

		aSystemFullID = aSelectedSystemLeafMap[0].get_data( 'FULLID' )
		self.theEntityList.clear_items( 0,-1 )

		#if aPrimitiveTypeString == 'All':
		if aEntityTypeString == 'All':
			self.listEntity( 'Variable', aSystemFullID )
			self.listEntity( 'Process', aSystemFullID )
		else:
			#self.listEntity( aPrimitiveTypeString, aSystemFullID )
			self.listEntity( aEntityTypeString, aSystemFullID )

	#def listEntity( self, aPrimitiveTypeString, aSystemFullID ):
	def listEntity( self, aEntityTypeString, aSystemFullID ):
		#aListPN = aPrimitiveTypeString + 'List'
		aListPN = aEntityTypeString + 'List'
		aListFullPN = convertFullIDToFullPN( aSystemFullID, aListPN ) 
		aEntityList = self.theMainWindow.theSession.theSimulator.getEntityProperty( createFullPNString( aListFullPN ) )

		for aEntityID in aEntityList:
			aListItem = gtk.GtkListItem( aEntityID )
            
			#if aPrimitiveTypeString == 'Variable':
			if aEntityTypeString == 'Variable':
				#aPrimitiveType = VARIABLE
				aEntityType = VARIABLE
			#elif aPrimitiveTypeString == 'Process':
			elif aEntityTypeString == 'Process':
				aEntityType = PROCESS

			aSystemPath = createSystemPathFromFullID( aSystemFullID )

			#aEntityFullPN = ( aPrimitiveType, aSystemPath, aEntityID, '' )
			aEntityFullPN = ( aEntityType, aSystemPath, aEntityID, '' )
#            aEntityFullPN = ( aPrimitiveType, aSystemPath, aEntityID, 'Value' )

			aListItem.set_data( 'FULLPN', aEntityFullPN)
#            aFullPNList = ( aEntityFullPN, )

			self.theEntityList.add( aListItem )
			aListItem.show()

	def selectEntity( self, aEntityList ):

		aSelectedEntityListItemList = aEntityList.get_selection()

		if len(aSelectedEntityListItemList) == 0 :
			return

		self.theSelectedFullPNList = []
		for aEntityListItem in aSelectedEntityListItemList:
			aEntityFullPN = aEntityListItem.get_data( 'FULLPN' )
			self.theSelectedFullPNList.append( aEntityFullPN )

		self.thePropertyWindow.theRawFullPNList = self.theSelectedFullPNList
		self.thePropertyWindow.setFullPNList()
		self.updateStatusBar()

	def selectPropertyName( self, aCList, row, column, event_obj ):

		self.theSelectedFullPNList = []
		for aRowNumber in aCList.selection:
			aPropertyName =  aCList.get_text( aRowNumber, 0 )
			aFullID = self.thePropertyWindow.theFullID()
			aFullPN = convertFullIDToFullPN( aFullID, aPropertyName )
			self.theSelectedFullPNList.append( aFullPN )
		self.updateStatusBar()
        
	def selectSystem( self, aTreeItemObj ):
		aFullID = aTreeItemObj.get_data('FULLID')
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
# PopupMenu -> GtkMenu
# ---------------------------------------------------------------
class PopupMenu( GtkMenu ):

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
		GtkMenu.__init__(self)
		self.theLoggerString = "add logger list"

		# ------------------------------------------
		# calls the constructor of super class
		# ------------------------------------------
		GtkMenu.__init__(self)

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
			self.theMenuItem[aPluginMap]= GtkMenuItem(aPluginMap)
			self.theMenuItem[aPluginMap].connect('activate', self.theParent.openNewPluginWindowByPopupMenu )
			self.theMenuItem[aPluginMap].set_name(aPluginMap)
			self.append( self.theMenuItem[aPluginMap] )
			if aMaxStringLength < len(aPluginMap):
				aMaxStringLength = len(aPluginMap)
			aMenuSize += 1
		self.append( gtk.GtkMenuItem() )

		# ------------------------------------------
		# adds creates logger
		# ------------------------------------------
		self.theMenuItem[self.theLoggerString]= GtkMenuItem(self.theLoggerString)
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
		GtkMenu.popup(self, pms, pmi, func, button, time)
		self.show_all(self)
	
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
