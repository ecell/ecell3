from gnome.canvas import *

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
