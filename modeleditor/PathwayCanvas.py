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
					gtk.gdk.Cursor( gtk.gdk.LEFT_SIDE ),
					gtk.gdk.Cursor( gtk.gdk.HAND2 ),
					gtk.gdk.Cursor( gtk.gdk.X_CURSOR) ]
		self.addHandlers()
		self.beyondcanvas=False
		self.zoomratio=0
		self.theRTMap={}
		

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
	
	def scrollTo(self, dx, dy, argum=None):
		if argum==None:
			x,y=self.theCanvas.get_scroll_offsets()
			self.theCanvas.scroll_to( x+dx, y+dy)
		else:
			self.theCanvas.scroll_to( dx, dy)


	def setZoomRatio( self, ppu):
		self.theCanvas.set_pixels_per_unit( ppu )
		self.zoomratio=ppu
		for aLabel, aText in self.theRTMap.iteritems():
			aText.set_property('size')
			

	def getZoomRatio( self):
		return self.zoomratio


	#--------------------------------
	#	ResizeableText methods
	#--------------------------------
	def registerText(self,aResizeableText,aLabel):
		self.theRTMap[aLabel]=aResizeableText
	
	def setLabelText(self,oldLabel,newLabel):
		if self.theRTMap.has_key(oldLabel):
			self.theRTMap[newLabel]=self.theRTMap[oldLabel]
			del self.theRTMap[oldLabel]
	
	def deregisterText(self,aLabel):
		if self.theRTMap.has_key(aLabel):
			del self.theRTMap[aLabel]



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
		
