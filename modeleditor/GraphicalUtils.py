from Constants import *
import gtk
import gnome.canvas

class GraphicalUtils:

	def __init__( self, aGtkWindow ):
		self.theGtkWindow = gtk.Window()
		self.theGdkWindow = self.theGtkWindow.window
		self.theColorMap = self.theGtkWindow.get_colormap()
		self.theStyle = self.theGtkWindow.get_style()
		self.testCanvas = gnome.canvas.Canvas()
		self.testCanvasRoot = self.testCanvas.root()
		self.testText=self.testCanvasRoot.add(gnome.canvas.CanvasText, text = 'Hello' )

	def getGdkColorByName( self, aColorName ):

		return self.theColorMap.alloc_color( aColorName )


	def getGdkColorByRGB( self, rgbArray ):
		aGdkColor = self.theStyle.white.copy()
		aGdkColor.red = rgbArray[0]
		aGdkColor.green = rgbArray[1]
		aGdkColor.blue = rgbArray[2]

		

	def getRRGByName( self, aColorName ):
		gdkColor = self.getGdkColorByName( aColorName )
		rgbArray = [ gdkColor.red, gdkColor.green, gdkColor.blue ]


	def getTextDimensions( self, aText ):
		self.testText.set_property('text',aText)
		width = self.testText.get_property('text-width')
		height = self.testText.get_property('text-height')
		return (height, width )


	def truncateTextToSize ( self, aText, aSize ):
		return aText
		
		
