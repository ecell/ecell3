#!/usr/bin/env python

from Window import *

from gtk import *
from ecell.ecssupport import *

import MainWindow

import string
import copy

# import test

class EntryListWindow(Window):

    def __init__( self, aMainWindow ):
        self.theSelectedFullPNList = []

        Window.__init__( self )
        self.addHandlers( { 'show_button_clicked'           : self.openNewPluginWindow,
                            'system_tree_selection_changed' : self.updateEntryList,
                            'entry_list_selection_changed'  : self.selectEntity } )

        self.theMainWindow = aMainWindow
        self.thePaletteWindow = aMainWindow.thePaletteWindow

        self.theSystemTree = self.getWidget( 'system_tree' )
        self.theEntryList = self.getWidget( 'entry_list' )
        self.theTypeOptionMenu = self.getWidget( 'type_optionmenu' )
        self.theStatusBar = self.getWidget( 'statusbar' )

        self.theTypeMenu = self.theTypeOptionMenu.get_menu()
        self.theTypeMenu.connect( 'selection-done', self.updateEntryList)
        aTypeMenuItemMap = self.theTypeMenu.children()
        aTypeMenuItemMap[0].set_data( 'LABEL', 'Substance' )
        aTypeMenuItemMap[1].set_data( 'LABEL', 'Reactor' )
        aTypeMenuItemMap[2].set_data( 'LABEL', 'All' )

        self.theSystemTree.show()
        self.theEntryList.show()

        aPManager = self.theMainWindow.thePluginManager
        self.thePropertyWindow = aPManager.createInstance( 'PropertyWindow',
                                                           [(4, '', '/', '')],'top_vbox' )

#        self.theSystemTree.select_item( -1 )

        aPropertyWindowTopVBox = self.thePropertyWindow['top_vbox']
        self['property_frame'].add( aPropertyWindowTopVBox )
        self.thePropertyWindow['propLerty_clist'].connect( 'select_row', self.selectPropertyName )
        
        self.update()

    def update( self ):
        self.theSystemTree.clear_items(0, 999)
        self.theEntryList.clear_items(0, 999)
        aRootSystemFullID = createFullID( 'System:/:/' )
        self.constructTree( self.theSystemTree, aRootSystemFullID )
        self.updateEntryList()
            
    def constructTree( self, aParentTree, aSystemFullID ):
        aLeaf = gtk.GtkTreeItem( label=aSystemFullID[ID] )
        aLeaf.set_data( 'FULLID', aSystemFullID )
        aLeaf.connect( 'select', self.selectSystem )
        aParentTree.append( aLeaf )
        aLeaf.show()

        aSystemListFullPN = convertFullIDToFullPN( aSystemFullID, 'SystemList' ) 
        aSystemList = self.theSession.theSimulator.getProperty( createFullIDString(aSystemListFullPN))
        if aSystemList != ():
            aTree = gtk.GtkTree()
            aLeaf.set_subtree( aTree )
            aLeaf.expand()

            for aSystemID in aSystemList:
                aSystemPath = createSystemPathFromFullID( aSystemFullID )
                aNewSystemFullID = ( SYSTEM, aSystemPath, aSystemID )
                self.constructTree( aTree, aNewSystemFullID )

    def updateEntryList( self, obj=None ):

        aSelectedSystemLeafMap = self.theSystemTree.get_selection()
        if aSelectedSystemLeafMap == [] :
            return
        
        aSelectedTypeMenuItem = self.theTypeMenu.get_active()
        aEntityTypeString = aSelectedTypeMenuItem.get_data( 'LABEL' )

        aSystemFullID = aSelectedSystemLeafMap[0].get_data( 'FULLID' )

        self.theEntryList.clear_items( 0,-1 )

        if aEntityTypeString == 'All':
            self.listEntity( 'Substance', aSystemFullID )
            self.listEntity( 'Reactor', aSystemFullID )
        else:
            self.listEntity( aEntityTypeString, aSystemFullID )

    def listEntity( self, aEntityTypeString, aSystemFullID ):
        aListPN = aEntityTypeString + 'List'
        aListFullPN = convertFullIDToFullPN( aSystemFullID, aListPN ) 
        aEntityList = self.theSession.theSimulator.getProperty( createFullIDString( aListFullPN ) )
        for aEntityID in aEntityList:
            aListItem = gtk.GtkListItem( aEntityID )
            
            if aEntityTypeString == 'Substance':
                aEntityType = SUBSTANCE
            elif aEntityTypeString == 'Reactor':
                aEntityType = REACTOR

            aSystemPath = createSystemPathFromFullID( aSystemFullID )

            aEntityFullPN = ( aEntityType, aSystemPath, aEntityID, '' )
#            aEntityFullPN = ( aEntityType, aSystemPath, aEntityID, 'Quantity' )

            aListItem.set_data( 'FULLPN', aEntityFullPN)
#            aFullPNList = ( aEntityFullPN, )

            self.theEntryList.add( aListItem )
            aListItem.show()

    def selectEntity( self, aEntryList ):
        aSelectedEntityListItemList = aEntryList.get_selection()

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
        aPluginName = self.thePaletteWindow.getSelectedPluginName()
        aPluginManager = self.theMainWindow.thePluginManager
        aPluginManager.createInstance( aPluginName,
                                       self.theSelectedFullPNList)
	

def mainQuit( obj, data ):
    gtk.mainquit()

def mainLoop():
    gtk.mainloop()

def main():
    aWindow = EntryListWindow( 'EntryListWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )    
    mainLoop()

def ecstest():
    aWindow = EntryListWindow( 'EntryListWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )    
    mainLoop()
    
if __name__ == "__main__":
    main()

if __name__ == "__ecstest__":
    ecstest()












