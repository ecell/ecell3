import gnome.canvas
from Constants import *

class ComplexShape:

	def __init__( self, anObject, aCanvas, x, y, width, height ):
		self.theCanvas = aCanvas
		self.parentObject = anObject
		self.graphUtils = self.parentObject.getGraphUtils()
		self.x = x
		self.y = y
		self.width = width
		self.height = height
		self.shapeMap = {}
		self.lastmousex = 0
		self.lastmousey = 0
		self.buttonpressed = False


	def show ( self ):
		self.theRoot = self.parentObject.theCanvas.getRoot()
		self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptorList()
		self.__sortByZOrder( self.shapeDescriptorList )
		for aDescriptor in self.shapeDescriptorList:
			if aDescriptor[SD_TYPE] == CV_RECT:

				self.createRectangle( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_ELL:
				self.createEllipse( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_TEXT:
				self.createText( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_LINE:
				self.createLine( aDescriptor )



	def addHandlers( self, canvasObject, aName ):
		canvasObject.connect('event', self.rect_event, aName )


	def createRectangle( self, aDescriptor ):
		rectSpec = aDescriptor[ SD_SPECIFIC ]
		relativeX1 = rectSpec[ RECT_RELX1 ] * self.width + rectSpec[ RECT_ABSX1 ]
		relativeY1 = rectSpec[ RECT_RELY1 ] * self.height + rectSpec[ RECT_ABSY1 ]
		relativeX2 = rectSpec[ RECT_RELX2 ] * self.width + rectSpec[ RECT_ABSX2 ]
		relativeY2 = rectSpec[ RECT_RELY2 ] * self.height + rectSpec[ RECT_ABSY2 ]
		(offsetx, offsety ) = self.parentObject.getAbsolutePosition()
		X1 = relativeX1 + offsetx
		X2 = relativeX2 + offsetx 
		Y1 = relativeY1 + offsety
		Y2 = relativeY2 + offsety 
		aGdkColor = self.getGdkColor( aDescriptor )

		aRect = self.theRoot.add( gnome.canvas.CanvasRect, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = aGdkColor, fill_color_gdk = aGdkColor )
		print aDescriptor[SD_NAME], X1, Y1, X2, Y2
		self.addHandlers( aRect, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aRect



	def createEllipse( self, aDescriptor ):
		pass


	def createLine( self, aDescriptor ):
		lineSpec = aDescriptor[SD_SPECIFIC]
		linePoints = lineSpec [ LINE_POINTS ]
		relativeX1 = linePoints[0] + linePoints[1] * self.width
		relativeY1 = linePoints[2] + linePoints[3] * self.height
		relativeX2 = linePoints[4] + linePoints[5] * self.width
		relativeY2 = linePoints[6] + linePoints[7] * self.height

		(offsetx, offsety ) = self.parentObject.getAbsolutePosition()
		X1 = relativeX1 + offsetx
		X2 = relativeX2 + offsetx
		Y1 = relativeY1 + offsety
		Y2 = relativeY2 + offsety
		print "x1, x2, y1, y2", X1, X2, Y1, Y2
		aGdkColor = self.getGdkColor( aDescriptor )
		aLine = self.theRoot.add( gnome.canvas.CanvasLine,points=[X1,Y1,X2,Y2], width_units=lineSpec[ LINE_WIDTH ], fill_color_gdk = aGdkColor )
		self.addHandlers( aLine, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aLine


	def createText( self, aDescriptor ):
		textSpec = aDescriptor[SD_SPECIFIC]
		relativeX1 = textSpec[TEXT_ABSX] + textSpec[TEXT_RELX] * self.width
		relativeY1 = textSpec[TEXT_ABSY] + textSpec[TEXT_RELY] * self.height

		(offsetx, offsety ) = self.parentObject.getAbsolutePosition()
		X1 = relativeX1 + offsetx
		Y1 = relativeY1 + offsety
		aGdkColor = self.getGdkColor( aDescriptor )
		aText = self.theRoot.add( gnome.canvas.CanvasText,x=X1,y=Y1, fill_color_gdk = aGdkColor, text = textSpec[TEXT_TEXT], anchor = gtk.ANCHOR_NW )
		self.addHandlers( aText, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aText


	def getGdkColor( self, aDescriptor ):
		aColorType = aDescriptor[ SD_COLOR ]
		if aColorType == SD_FILL:
			queryProp = OB_FILL_COLOR
		elif aColorType == SD_OUTLINE:
			queryProp = OB_OUTLINE_COLOR
		elif aColorType == SD_TEXT:
			queryProp = OB_TEXT_COLOR

		anRGBColor = self.parentObject.getProperty( queryProp )

		return self.graphUtils.getGdkColorByRGB( anRGBColor )



	def delete( self ):
		pass


	def __sortByZOrder ( self, desclist ):
		fn = lambda x,y: (x[SD_Z]<y[SD_Z] ) - ( y[SD_Z]<x[SD_Z])

		desclist.sort(fn)


	def leftClick( self, shapeName, x, y ):
		# usually select
		if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_SYSTEM_CANVAS:
			self.parentObject.addItem( x, y)
		else:
			self.parentObject.doSelect()


	def rightClick ( self, shapeName, x, y ):
		#usually show menu
		self.parentObject.showMenu()

	def mouseDrag( self, shapeName, deltax, deltay ):
		#decide whether resize or move or draw arrow
		pass

	def  doubleClick( self, shapeName ):
		self.parentObject.popupEditor()

	def getShapeDescriptor( self, shapeName ):
		return self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptor( shapeName )


	def rect_event( self, *args ):

		event = args[1]
		item = args[0]
		shapeName = args[2]

		if event.type == gtk.gdk.BUTTON_PRESS:
			if event.button == 1:
				self.lastmousex = event.x
				self.lastmousey = event.y
				self.buttonpressed = True
				self.leftClick( shapeName, event.x, event.y )
			elif event.button == 3:
				self.rightClick(shapeName, event.x, event.y )

		elif event.type == gtk.gdk.BUTTON_RELEASE:
			if event.button == 1:
				self.buttonpressed = False

		elif event.type == gtk.gdk.MOTION_NOTIFY:
			if not self.buttonpressed:
				return
			oldx = self.lastmousex
			oldy = self.lastmousey
			self.lastmousex = event.x
			self.lastmousey = event.y
			deltax = event.x - oldx
			deltay = event.y - oldy
			self.mouseDrag( shapeName, deltax, deltay )
			
		elif event.type == gtk.gdk._2BUTTON_PRESS:
			if event.button == 1:
				self.doubleClick( shapeName )

