#!/usr/bin/env python

from Window import *

from gtk import *
from ecssupport import *

# from MainWindow import *
import MainWindow

import string
import copy

# import test

class EntryListWindow(Window):

    def __init__( self, aMainWindow, aSimulator ):

        Window.__init__( self, 'EntryListWindow.glade' )
        self.addHandlers( { 'show_button_clicked' : self.openNewPluginWindow
                            } )

        self.theSimulator = aSimulator
        self.theMainWindow = aMainWindow
        self.thePaletteWindow = aMainWindow.thePaletteWindow
        self.theSystemTree = self.getWidget( 'system_tree' )
        self.theEntryList = self.getWidget( 'entry_list' )
        self.theTypeOptionMenu = self.getWidget( 'type_optionmenu' )
        self.theStatusBar = self.getWidget( 'statusbar' )

        self.theTypeMenu = self.theTypeOptionMenu.get_menu()
        aTypeMenuItemMap = self.theTypeMenu.children()
        aTypeMenuItemMap[0].connect( 'activate', self.changeEntityType,
                                     'Substance' )
        aTypeMenuItemMap[1].connect( 'activate', self.changeEntityType,
                                     'Reactor' )
        aTypeMenuItemMap[2].connect( 'activate', self.changeEntityType,
                                     'All' )

        self.theSelectedEntityType = 'Substance'
        self.theSelectedSystemFullID = ''
        self.theSelectedEntityList = []

        self.theSystemTree.show()
        self.theEntryList.show()

        aRootSystemFullID = FullID( 'System:/:/' )
        self.constructTree( self.theSystemTree, aRootSystemFullID )

    def constructTree( self, aParentTree, aSystemFullID ):
        aLeaf = gtk.GtkTreeItem( label=aSystemFullID[ID] )
        aLeaf.connect( 'select', self.changeSystem, aSystemFullID )
        aParentTree.append( aLeaf )
        aLeaf.show()

        aSystemListFullPN = FullIDToFullPropertyName( aSystemFullID, 'SystemList' ) 
        aSystemList = self.theSimulator.getProperty( aSystemListFullPN )
        if aSystemList != ():
            aTree = gtk.GtkTree()
            aLeaf.set_subtree( aTree )
            aLeaf.expand()

            for aSystemID in aSystemList:
                aSystemPath = self.createSystemPathFromFullID( aSystemFullID )
                aNewSystemFullID = ( SYSTEM, aSystemPath, aSystemID )
                self.constructTree( aTree, aNewSystemFullID )

    def changeEntityType( self, menu_item_obj, aEntityType):
        self.theSelectedEntityType = aEntityType
        self.updateEntryList( aEntityType=aEntityType )

    def changeSystem( self, a, aSystemFullID):
        self.theSelectedSystemFullID = aSystemFullID
        self.updateEntryList( aSystemFullID=aSystemFullID )

    def updateEntryList( self, aEntityType='default', aSystemFullID='default' ):
        if aEntityType == 'default':
            aEntityType = self.theSelectedEntityType
        if aSystemFullID == 'default':
            aSystemFullID = self.theSelectedSystemFullID

        self.theEntryList.clear_items( 0,-1 )

        if aEntityType == 'All':
            self.listEntity( 'Substance', aSystemFullID )
            self.listEntity( 'Reactor', aSystemFullID )
        else:
            self.listEntity( aEntityType, aSystemFullID )

    def listEntity( self, aEntityTypeString, aSystemFullID ):
        aListPN = aEntityTypeString + 'List'
        aListFullPN = FullIDToFullPropertyName( aSystemFullID, aListPN ) 
        aEntityList = self.theSimulator.getProperty( aListFullPN )
        for aEntityID in aEntityList:
            aListItem = gtk.GtkListItem( aEntityID )

            aSystemPath = self.createSystemPathFromFullID( aSystemFullID )
            aListItem.connect( 'select', self.selectEntity, aEntityTypeString,
                               aSystemPath, aEntityID )
            self.theEntryList.add( aListItem )
            aListItem.show()

    def selectEntity( self, a, aEntityTypeString, aSystemPath, aID ):
        if aEntityTypeString == 'Substance':
            aEntityType = SUBSTANCE
        elif aEntityTypeString == 'Reactor':
            aEntityType = REACTOR

        self.theSelectedEntityList = (( aEntityType, aSystemPath, aID, 'Quantity' ), )
        
    def openNewPluginWindow( self, button_obj ) :
        
        aPluginList = self.thePaletteWindow.get_data( 'plugin_list' )
        for plugin_name in aPluginList :
            aButton = self.thePaletteWindow.get_data( plugin_name )
            if aButton.get_active() :
                aPluginName = plugin_name

        print self.theSelectedEntityList
        self.theMainWindow.thePluginManager.createInstance( aPluginName,
                   self.theSimulator, self.theSelectedEntityList )
                                                            
    def createSystemPathFromFullID( self, aSystemFullID ):
        if aSystemFullID[SYSTEMPATH] == '/':
            if aSystemFullID[ID] == '/':
                aNewSystemPath = '/'
            else:
                aNewSystemPath = '/' + aSystemFullID[ID]
        else:
            aNewSystemPath = aSystemFullID[SYSTEMPATH] + '/' +\
                             aSystemFullID[ID]
        return aNewSystemPath


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












