#!/usr/bin/env python

from Interface import *
from ecssupport import *

#TARGET_STRING = 0
#TARGET_ROOTWIN = 1

#target = [ ('STRING', 0, TARGET_STRING),
#           ('text/plain', 0, TARGET_STRING),
#           ('application/x-rootwin-drop', 0, TARGET_ROOTWIN) ]

class NumericWindow( Interface ):

    def __init__( self, fpn ):
        self.theGladeFileName = 'NumericWindow.glade'
        Interface.__init__( self, self.theGladeFileName, fpn )
        self.addHandler('input', self.inputValue)
        self.initInterface( fpn )
 #       self["fqpp_label"].drag_dest_set( gtk.DEST_DEFAULT_ALL, target, GDK.ACTION_COPY)
#        self["fqpp_label"].drag_source_set( GDK.BUTTON1_MASK, target, GDK.ACTION_COPY|GDK.ACTION_MOVE )

    def initInterface( self, fpn ):
        fpnobj = FullPropertyName(fpn)
        self["fqpp_label"].set_text(fpnobj['ID'])
        self["spinbutton"].set_text('1.00000')

    def inputValue( self, obj, n ):
        aNumberString =  obj.get_text()
        aNumber = string.atof( aNumberString )
        print aNumberString

### test code

def mainLoop():

    aFPN1 = 'Substance:/CELL/CYTOPLASM:ATP:Quantity'
    aWindow1 = NumericWindow( aFPN1 )
    aFPN2 = 'Substance:/CELL/CYTOPLASM:ADP:Quantity'
    aWindow2 = NumericWindow( aFPN2 )

#    aWindow2 = NumericWindow( 'Substance:/CELL/CYTOPLASM:ATP' )
#    aWindow3 = NumericWindow( 'Reactor:/CELL/CYTOPLASM:ATPase-0' )
#    aWindow4 = NumericWindow( 'System:/CELL:CYTOPLASM' )
#    aWindow5 = NumericWindow( 'Substance:/CELL/CYTOPLASM:ATP|Quantity,Substance:/CELL/CYTOPLASM:ADP|Quantity' )
#    aWindow6 = NumericWindow( 'Substance:/CELL/CYTOPLASM:ATP|Quantity,Substance:/CELL/CYTOPLASM:ATPase' )
    
    gtk.mainloop()

def main():
    mainLoop()

if __name__ == "__main__":
    main()

















