from gnome.canvas import *
import gtk.gdk

class PathwayCanvas( Canvas ):
	
	def __init__( self, parentWindow, aCanvas):
		self.theParentWindow = parentWindow
		self.theCanvas = aCanvas
		self.theCanvasRoot = self.theCanvas.root()
		aStyle = aCanvas.get_style().copy()
		aGdkWindow = aCanvas.window
		aColorMap = aGdkWindow.get_colormap()
		whiteColor = aColorMap.alloc_color("white")
		aStyle.bg[0] = whiteColor
		aCanvas.set_style( aStyle )
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
					gtk.gdk.Cursor( gtk.gdk.LEFT_SIDE ) ]

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

	def setZoomRatio( self, ppu):
		self.theCanvas.set_pixels_per_unit( ppu )
