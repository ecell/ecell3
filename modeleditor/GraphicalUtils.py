

from Constants import *
import gtk
import gnome.canvas
import Numeric as n

class GraphicalUtils:

	def __init__( self, aGtkWindow ):
		self.theGtkWindow = gtk.Window()
		self.theGdkWindow = self.theGtkWindow.window
		self.theColorMap = self.theGtkWindow.get_colormap()
		self.theStyle = self.theGtkWindow.get_style()
		self.testCanvas = gnome.canvas.Canvas()
		self.testCanvasRoot = self.testCanvas.root()
		self.testText=self.testCanvasRoot.add(gnome.canvas.CanvasText, text = 'Hello' )
		self.opMatrix= n.array(((0,0,1,0,-1,0,0,0),(-1,0,0,0,0,0,1,0),(0,0,0,1,0,-1,0,0),(0,-1,0,0,0,0,0,1)))

	def getGdkColorByName( self, aColorName ):

		return self.theColorMap.alloc_color( aColorName )


	def getGdkColorByRGB( self, rgbArray ):
		aGdkColor = self.theStyle.white.copy()
		aGdkColor.red = rgbArray[0]
		aGdkColor.green = rgbArray[1]
		aGdkColor.blue = rgbArray[2]
		return aGdkColor
		

	def getRRGByName( self, aColorName ):
		gdkColor = self.getGdkColorByName( aColorName )

		return [ gdkColor.red, gdkColor.green, gdkColor.blue ]



	def getTextDimensions( self, aText ):
		self.testText.set_property('text',aText)
		width = self.testText.get_property('text-width')
		height = self.testText.get_property('text-height')
		return (height, width )


	def truncateTextToSize ( self, aText, aSize ):
		return aText
		
	
	def areOverlapping( self, rect1, rect2 ):
		b= n.concatenate( (rect1,rect2))
		return  n.sometrue(n.less(n.dot(self.opMatrix,b),0))

