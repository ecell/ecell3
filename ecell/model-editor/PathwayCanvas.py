#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
import gtk.gdk
try:
    import gnomecanvas
except:
    import gnome.canvas as gnomecanvas

class PathwayCanvas( gnomecanvas.Canvas ):
    
    def __init__( self, parentWindow, aCanvas):
        self.theParentWindow = parentWindow
        self.theCanvas = aCanvas
        self.theCanvasRoot = self.theCanvas.root()
        aStyle = aCanvas.get_style().copy()
        #aGdkWindow = aCanvas.window
        #aColorMap = aGdkWindow.get_colormap()
        #whiteColor = aColorMap.alloc_color("white")
        #aStyle.bg[0] = whiteColor
        #aCanvas.set_style( aStyle )
        self.theCursorList = [ \
                    gtk.gdk.Cursor( gtk.gdk.TOP_LEFT_ARROW ),
                    gtk.gdk.Cursor( gtk.gdk.FLEUR ),
                    gtk.gdk.Cursor( gtk.gdk.PLUS ),
                    gtk.gdk.Cursor( gtk.gdk.TOP_LEFT_CORNER ),
                    gtk.gdk.Cursor( gtk.gdk.TOP_SIDE ),
                    gtk.gdk.Cursor( gtk.gdk.TOP_RIGHT_CORNER ),
                    gtk.gdk.Cursor( gtk.gdk.RIGHT_SIDE ),
                    gtk.gdk.Cursor( gtk.gdk.BOTTOM_RIGHT_CORNER ),
                    gtk.gdk.Cursor( gtk.gdk.BOTTOM_SIDE ),
                    gtk.gdk.Cursor( gtk.gdk.BOTTOM_LEFT_CORNER ),
                    gtk.gdk.Cursor( gtk.gdk.LEFT_SIDE ),
                    gtk.gdk.Cursor( gtk.gdk.HAND2 ),
                    gtk.gdk.Cursor( gtk.gdk.X_CURSOR) ]
        self.addHandlers()
        self.beyondcanvas=False
        self.recentScroll = False
        self.zoomratio=1
        self.theRTMap={}
        self.lastposx=0
        self.lastposy=0

    def setCursor( self, aCursorType ):
        aCursor = self.theCursorList[ aCursorType ]
        self.theCanvas.window.set_cursor( aCursor )

        
    def getParentWindow( self ):
        return self.theParentWindow
        
    def setLayout( self, aLayout ):
        self.theLayout = aLayout
        
    def getLayout( self ):
        return self.theLayout

    def getCanvas( self ):
        return self.theCanvas

    def getRoot( self ):
        return self.theCanvasRoot
    
    def setSize( self, scrollRegion ):
        
        self.theCanvas.set_scroll_region( scrollRegion[0], scrollRegion[1], scrollRegion[2], scrollRegion[3] )
        

    def getSize(self):
        return self.theCanvas.get_size()

    def setLastCursorPos(self,x,y):
        self.lastposx=x
        self.lastposy=y

    def getLastCursorPos(self):
        return self.lastposx,self.lastposy
    
    def scrollTo(self, dx, dy, argum=None):
        dx*=self.zoomratio
        dy*=self.zoomratio
        if argum==None:
            x,y=self.theCanvas.get_scroll_offsets()
            self.theCanvas.scroll_to(int(x+dx) , int(y+dy))
            self.setLastCursorPos(x+dx,y+dy)
        else:
            self.theCanvas.scroll_to( int(dx), int(dy))
        self.recentScroll = True
        
    def getRecentScroll( self ):
        if self.recentScroll:
            self.recentScroll = False
            return True
        return False
            


    def setZoomRatio( self, ppu):
        lastx,lasty=self.theCanvas.get_scroll_offsets()
        self.theCanvas.set_pixels_per_unit( ppu )
        self.zoomratio=ppu
        for aResizeableText, aText in self.theRTMap.iteritems():
            aResizeableText.set_property('size')
        self.scrollTo(lastx,lasty,'attach')
            

    def getZoomRatio( self):
        return self.zoomratio


    #--------------------------------
    #   ResizeableText methods
    #--------------------------------
    def registerText(self,aResizeableText,aText):
        self.theRTMap[aResizeableText]=aText
    
    #def setLabelText(self,oldLabel,newLabel):
    #   if self.theRTMap.has_key(oldLabel):
    #       self.theRTMap[newLabel]=self.theRTMap[oldLabel]
    #       del self.theRTMap[oldLabel]
    def setLabelText(self,aResizeableText,newText):
        if self.theRTMap.has_key(aResizeableText):
            self.theRTMap[aResizeableText]=newText
    
    #def deregisterText(self,aLabel):
    #   if self.theRTMap.has_key(aLabel):
    #       del self.theRTMap[aLabel]
    def deregisterText(self,aResizeableText):
        if self.theRTMap.has_key(aResizeableText):
            del self.theRTMap[aResizeableText]


    def addHandlers( self):
        self.theCanvas.connect('event', self.canvas_event)

    def canvas_event( self, *args):
        event=args[1]
        if event.type == gtk.gdk.ENTER_NOTIFY:
            self.beyondcanvas=False

        elif event.type == gtk.gdk.LEAVE_NOTIFY:
            self.beyondcanvas=True

    def getBeyondCanvas(self):
        return self.beyondcanvas
        
