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

    def __init__( self, gladefile ):

        Window.__init__( self, gladefile )
        self.addHandlers( {
            # 'selection_changed': self.showList 
            # 'select_child': self.showList
            } )

        self.theTree = self.getWidget( 'tree10' )
        self.theList = self.getWidget( 'list1' )
        self.theOptionMenu = self.getWidget( 'optionmenu3' )

    # def optionMenu
        self.theOpmenu = 'Substance'
        self.theMenu = gtk.GtkMenu()

        ## option menu tmp ##
        sText = 'Substance'
        aMenuItem = gtk.GtkMenuItem( label=sText )
        self.theMenu.append( aMenuItem )
        aMenuItem.show()            
        self.theOptionMenu.set_menu( self.theMenu )
        aMenuItem.connect( 'activate', self.showListPre, sText )

        rText = 'Reactor'
        aMenuItem = gtk.GtkMenuItem( label=rText )
        self.theMenu.append( aMenuItem )
        aMenuItem.show()            
        self.theOptionMenu.set_menu( self.theMenu )
        aMenuItem.connect( 'activate', self.showListPre, rText )


        s = MainWindow.simulator()
        
        self.initialize()

        self.theTree.show()
        self.theList.show()
        self.theOptionMenu.show()

    def initialize( self ):

        s = MainWindow.simulator()

        self.theRootTreeItem = gtk.GtkTreeItem( label='/')
        self.theTree.append( self.theRootTreeItem )
        self.theRootTreeItem.show()
        self.theRootTreeItem.connect( 'select', self.showListTre, s.theRootSystem )

        self.constructTree( self.theRootTreeItem, s.theRootSystem  )

    def constructTree( self, parent, path ):

        aList = self.toList( path )
        if aList != ():
            aSubTree = gtk.GtkTree()
            parent.set_subtree( aSubTree )
            parent.expand()

            for x in aList:
                self.aNewLeaf = gtk.GtkTreeItem( label=x )
                aSubTree.append( self.aNewLeaf )
                self.aNewLeaf.show()            
                self.aNewLeaf.connect( 'select', self.showListTre, self.toList2(path,x) )
                self.constructTree( self.aNewLeaf, self.toList2(path,x) )

    def toList( self, path ):
        aList = path['SystemList']
        return aList

    def toList2( self, path, itemname ):
        abc = path[itemname]
        return abc

    def appendItem( self, item ):
        self.aListItem = gtk.GtkListItem( item )
        self.theList.add( self.aListItem )
        self.aListItem.show()# exit

    def appendItemList( self, list ):
        for x in list:
            self.appendItem( x )

    def doList( self, list ):
        self.theList.clear_items( 0,-1 )
        self.appendItemList( list )

    def showListPre( self, b, opmenu ):                   #OptionMenu activate
        self.theB = b
        self.theOpmenu = opmenu
        self.showList( b, self.theSystemname, opmenu )

    def showListTre( self, a, systemname ):               #TreeItem selected
        self.theSystemname = systemname
        self.theA = a
        self.showList( a, systemname, self.theOpmenu )
        
    def showList( self, a, systemname, opmenu='substance' ):
        if opmenu == 'Substance':
            aListType = 'SubstanceList'
        if opmenu == 'Reactor':
            aListType = 'ReactorList'
        aList = systemname[aListType]
        # self.theA = a
        # self.theSystemname = systemname
        self.doList( aList )

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












