import gnome.canvas

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


	def show ( self ):
		self.theRoot = self.parentObject.theCanvas.getRoot()
		self.shapeDescriptorList = self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST )
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


	def addHandlers( self, canvasObject, aFunction ):
		pass


	def createRectangle( self, aDescriptor ):
		relativeX1 = aDescriptor[ RECT_RELX ] * self.width + aDescriptor[ RECT_ABSX ]
		relativeY1 = aDescriptor[ RECT_RELY ] * self.height + aDescriptor[ RECT_ABSY ]
		relativeX2 = relativeX1 + aDescriptor[ RECT_RELWIDTH ] * self.width + aDescriptor[ RECT_ABSWIDTH ]
		relativeY2 = relativeY1 + aDescriptor[ RECT_RELHEIGHT ] * self.height + aDescriptor[ RECT_ABSHEIGHT ]
		(offsetx, offsety ) = self.parentObject.getAbsolutePosition()
		X1 = relativeX1 + offsetx
		X2 = relativeX2 + offsetx
		Y1 = relativeY1 + offsety
		Y2 = relativeY2 + offsety
		aFunction = aDescriptor[ SD_FUNCTION ]
		anRGBColor = aDescriptor[ SD_COLOR ]
		aGdkColor = self.graphUtils.getGdkColorByRGB( anRGBColor )
		aRect = self.theRoot.add( gnome.canvas.CanvasRect, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = aGdkColor, fill_color_gdk = aGdkColor )
		self.addHandlers( aRect, aFunction)
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aRect


	def createEllipse( self, aDescriptor ):
		pass


	def createLine( self, aDescriptor ):
		pass


	def createText( self, aDescriptor ):
		pass



	def delete( self ):
		pass


	def __sortByZOrder ( self, desclist ):
		fn = lambda x,y: (x[SD_Z]<y[SD_Z] ) - ( y[SD_Z]<x[SD_Z])

		desclist.sort(fn)
