#!/usr/bin/env python

import string

import gtk
import gnome.ui
import GDK
import libglade

import PekinTest

class Window:

    def __init__( self, gladefile=None, root=None ):
        self.widgets = libglade.GladeXML( filename=gladefile, root=root )

    def addHandlers( self, handlers ):
        self.widgets.signal_autoconnect( handlers )
        
    def addHandler( self, name, handler, *args ):
        self.widgets.signal_connect( name, handler, args )

    def getWidget( self, key ):
        return self.widgets.get_widget( key )

    def __getitem__( self, key ):
        return self.widgets.get_widget( key )

class EntryListWindow(Window):

    def __init__( self, gladefile ):

        theHandlerMap = {
           'list_show': self.list_show,
           'on_tree10_selection_changed': self.ShowEntry
            }

        self.theNewLeaf = {}


        Window.__init__( self, gladefile )
        self.addHandlers( theHandlerMap )

        self.theTree = self.getWidget( 'tree10' )
        self.theSubTree = gtk.GtkTree()

        self.theList = self.getWidget( 'list1' )

        self.theLeaf = gtk.GtkTreeItem('/')
        self.theTree.append( self.theLeaf )

        self.theLeaf.set_subtree(self.theSubTree)

        for x in range(len(PekinTest.testdic['System'])):
            self.theNewLeaf[x] = gtk.GtkTreeItem(PekinTest.testdic['System'][x])
            print self.theNewLeaf[x]
            self.theSubTree.append( self.theNewLeaf[x])

#        print PekinTest.testdic['System']
#        self.theLeaf = PekinTest.testdic['System']

#        self.theLeaf.set_subtree(self.theSubTree)
#        self.theNewLeaf = gtk.GtkTreeItem( 'Second level leaf!' )
#        self.theSubTree.append( self.theNewLeaf )

    def doList( self ):
        self.appendItemList( ['kem', 'nori', 'pe'] )

        for x in range(len(PekinTest.testdic['System'])):
            self.theNewLeaf[x].show()

        self.theLeaf.show()
        self.theTree.show()

        self.theList.show()

    def appendItem( self, item ):
        self.aListItem = gtk.GtkListItem( item )
        self.theList.add( self.aListItem )
        self.aListItem.show()

    def appendItemList( self, list ):
        for x in list:
            self.appendItem( x )

    def list_show( self,obj ):
        # if 
        self.theList.clear_items()
        self.doList()
    

    def ShowEntry( self,obj ):
       print 'showentry'
       
def mainQuit( obj, data ):
    print obj,data
    gtk.mainquit()

def mainLoop():
    gtk.mainloop()

def main():
    aWindow = EntryListWindow( 'EntryListWindow.glade' )
    aWindow.addHandler( 'gtk_main_quit', mainQuit )    
    aWindow.doList()
    mainLoop()

if __name__ == "__main__":
    main()

