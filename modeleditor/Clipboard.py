import gtk
import gtk.gdk


class Clipboard:

    def __init__( self ):
        self.theClipboard = gtk.Clipboard( gtk.gdk.display_get_default(), "CLIPBOARD" )
        self.isChanged = True
        
    def copyToClipboard( self, aText ):
        self.theClipboard.set_with_data( [ ( 'STRING', 0, 0 ), ('TEXT',0,0) ], self.__getFunc, self.__clearFunc, aText )
        self.isChanged = False
        
    def pasteFromClipboard( self ):
        return self.theClipboard.wait_for_text()
        
    def isClipboardChanged( self ):
        return self.isChanged

    def __clearFunc( self, a, b ):
        self.isChanged = True
        
        
    def __getFunc( self, aClipboard, aSelectionData, a, aText ):
        aSelectionData.set_text( aText, -1 )

        
