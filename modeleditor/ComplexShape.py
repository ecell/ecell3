import gnome.canvas
from Constants import *
from Utils import *


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
		self.isSelected = False
		for aDescriptor in self.shapeDescriptorList:
			if aDescriptor[SD_TYPE] == CV_RECT:
				self.createRectangle( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_ELL:
				self.createEllipse( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_TEXT:
				self.createText( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_LINE:
				self.createLine( aDescriptor )
		self.isSelected = False


	def delete( self ):
		pass

	def selected( self ):
		self.isSelected = True

	def unselected( self ):
		self.isSelected = False

	def outlineColorChanged( self ):
		# find shapes with outline color
		anRGB = copyValue( self.parentObject.getProperty( OB_OUTLINE_COLOR ) )
		if self.isSelected:
			for i in range(0,3):
				anRGB[i] = 32768 + anRGB[i]
		for aDescriptor in self.shapeDescriptorList:
			if aDescriptor[ SD_COLOR ] == SD_OUTLINE:

				aColor = self.graphUtils.getGdkColorByRGB( anRGB )
				if aDescriptor[SD_TYPE] == CV_RECT:
					self.changeRectColor( aDescriptor[ SD_NAME ] , aColor )
				elif aDescriptor[SD_TYPE] == CV_ELL:
					self.changeEllipseColor( aDescriptor[ SD_NAME ] , aColor )
				elif aDescriptor[SD_TYPE] == CV_LINE:
					self.changeLineColor( aDescriptor[ SD_NAME ] , aColor )

	
	def move( self, deltax, deltay ):

		for aDescriptor in self.shapeDescriptorList:
			if aDescriptor[SD_TYPE] == CV_RECT:
				self.moveRectangle( aDescriptor, deltax, deltay )
			elif aDescriptor[SD_TYPE] == CV_ELL:
				self.moveEllipse( aDescriptor, deltax, deltay  )
			elif aDescriptor[SD_TYPE] == CV_TEXT:
				self.moveText( aDescriptor, deltax, deltay  )
			elif aDescriptor[SD_TYPE] == CV_LINE:
				self.moveLine( aDescriptor, deltax, deltay  )

	def resize( self, deltawidth, deltaheight ):
		self.width += deltawidth
		self.height += deltaheight
		for aDescriptor in self.shapeDescriptorList:
			if aDescriptor[SD_TYPE] == CV_RECT:
				self.resizeRectangle( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_ELL:
				self.resizeEllipse( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_TEXT:
				self.resizeText( aDescriptor )
			elif aDescriptor[SD_TYPE] == CV_LINE:
				self.resizeLine( aDescriptor )
		

	def calculateRectCorners( self, aDescriptor ):
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
		return ( X1, X2, Y1, Y2 )

	def createRectangle( self, aDescriptor ):
		( X1, X2, Y1, Y2 ) = self.calculateRectCorners( aDescriptor )
		aGdkColor = self.getGdkColor( aDescriptor )

		aRect = self.theRoot.add( gnome.canvas.CanvasRect, x1=X1, y1=Y1, x2=X2, y2=Y2, outline_color_gdk = aGdkColor, fill_color_gdk = aGdkColor )
		self.addHandlers( aRect, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aRect


	def resizeRectangle( self, aDescriptor ):
		( X1, X2, Y1, Y2 ) = self.calculateRectCorners( aDescriptor )
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		aShape.set_property( 'x1', X1 )
		aShape.set_property( 'y1', Y1 )
		aShape.set_property( 'x2', X2 )
		aShape.set_property( 'y2', Y2 )


	def changeRectColor ( self, shapeName, aColor ):
		aShape = self.shapeMap[ shapeName  ]
		aShape.set_property('outline_color_gdk', aColor )
		aShape.set_property('fill_color_gdk', aColor )				



	def createEllipse( self, aDescriptor ):
		pass

	def calculateEllipseCorners( self, aDescriptor ):
		pass

	def resizeEllipse( self, aDescriptor ):
		( X1, X2, Y1, Y2 ) = self.calculateEllipseCorners( aDescriptor )
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		aShape.set_property( 'x1', X1 )
		aShape.set_property( 'y1', Y1 )
		aShape.set_property( 'x2', X2 )
		aShape.set_property( 'y2', Y2 )


	def changeEllipseColor ( self, shapeName, aColor ):
		aShape = self.shapeMap[  shapeName ] 
		aShape.set_property('outline_color_gdk', aColor )
		aShape.set_property('fill_color_gdk', aColor )


	def calculateLineDimensions( self, aDescriptor ):
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
		return ( X1, X2, Y1, Y2 )


	def createLine( self, aDescriptor ):
		lineSpec = aDescriptor[SD_SPECIFIC]
		( X1, X2, Y1, Y2 ) = self.calculateLineDimensions( aDescriptor )
		aGdkColor = self.getGdkColor( aDescriptor )
		aLine = self.theRoot.add( gnome.canvas.CanvasLine,points=[X1,Y1,X2,Y2], width_units=lineSpec[ LINE_WIDTH ], fill_color_gdk = aGdkColor )
		self.addHandlers( aLine, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aLine


	def resizeLine( self, aDescriptor ):
		( X1, X2, Y1, Y2 ) = self.calculateLineDimensions( aDescriptor )
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		aShape.set_property( 'points', [X1,Y1,X2,Y2] )


	def changeLineColor ( self, shapeName, aColor ):
		aShape = self.shapeMap[  shapeName ] 
		aclr = aShape.get_property('fill_color_gdk')
		aShape.set_property('fill_color_gdk', aColor )
		aclr = aShape.get_property('fill_color_gdk')

	def calculateTextDimensions( self, aDescriptor ):
		textSpec = aDescriptor[SD_SPECIFIC]
		relativeX1 = textSpec[TEXT_ABSX] + textSpec[TEXT_RELX] * self.width
		relativeY1 = textSpec[TEXT_ABSY] + textSpec[TEXT_RELY] * self.height

		(offsetx, offsety ) = self.parentObject.getAbsolutePosition()
		X1 = relativeX1 + offsetx
		Y1 = relativeY1 + offsety
		return (X1, Y1 )

	def createText( self, aDescriptor ):
		textSpec = aDescriptor[SD_SPECIFIC]
		(X1, Y1) = self.calculateTextDimensions( aDescriptor )
		aGdkColor = self.getGdkColor( aDescriptor )
		aText = self.theRoot.add( gnome.canvas.CanvasText,x=X1,y=Y1, fill_color_gdk = aGdkColor, text = textSpec[TEXT_TEXT], anchor = gtk.ANCHOR_NW )
		self.addHandlers( aText, aDescriptor[ SD_NAME ] )
		self.shapeMap[ aDescriptor[ SD_NAME ] ] = aText

	def resizeText( self, aDescriptor ):
		#by default text cannot be resized, it defines size 
		(x1, y1) = self.calculateTextDimensions( aDescriptor )
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		aShape.set_property( 'x', x1 )
		aShape.set_property( 'y', y1 )


	def moveRectangle( self, aDescriptor, deltax, deltay ):
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		x1 = aShape.get_property( 'x1' )
		y1 = aShape.get_property( 'y1' )
		x2 = aShape.get_property( 'x2' )
		y2 = aShape.get_property( 'y2' )
		x1 += deltax
		y1 += deltay
		x2 += deltax
		y2 += deltay
		aShape.set_property( 'x1', x1 )
		aShape.set_property( 'y1', y1 )
		aShape.set_property( 'x2', x2 )
		aShape.set_property( 'y2', y2 )


	def moveEllipse( self, aDescriptor, deltax, deltay ):
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		x1 = aShape.get_property( 'x1' )
		y1 = aShape.get_property( 'y1' )
		x2 = aShape.get_property( 'x2' )
		y2 = aShape.get_property( 'y2' )
		x1 += deltax
		y1 += deltay
		x2 += deltax
		y2 += deltay
		aShape.set_property( 'x1', x1 )
		aShape.set_property( 'y1', y1 )
		aShape.set_property( 'x2', x2 )
		aShape.set_property( 'y2', y2 )


	def moveLine( self, aDescriptor, deltax, deltay ):
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		(x1, y1, x2, y2 ) = aShape.get_property( 'points' )
		x1 += deltax
		y1 += deltay
		x2 += deltax
		y2 += deltay
		aShape.set_property( 'points', (x1, y1, x2, y2) )

	def moveText( self, aDescriptor, deltax, deltay ):
		aShape = self.shapeMap[ aDescriptor[ SD_NAME ] ]
		x = aShape.get_property( 'x' )
		y = aShape.get_property( 'y' )
		x += deltax
		y += deltay
		aShape.set_property( 'x', x )
		aShape.set_property( 'y', y )
		

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



	def __sortByZOrder ( self, desclist ):
		fn = lambda x, y: ( x[SD_Z] < y[SD_Z] ) - ( y[SD_Z] < x[SD_Z] )

		desclist.sort(fn)


	def leftClick( self, shapeName, x, y ):
		# usually select
		if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_SYSTEM_CANVAS:
			self.parentObject.addItem( x, y )
		else:
			self.parentObject.doSelect()


	def rightClick ( self, shapeName, x, y ):
		# usually show menu
		self.parentObject.showMenu()


	def mouseDrag( self, shapeName, deltax, deltay, origx, origy ):
		# decide whether resize or move or draw arrow
		if self.getShapeDescriptor(shapeName)[SD_FUNCTION] == SD_OUTLINE:

			self.parentObject.outlineDragged( deltax, deltay, origx, origy )
		else:
			self.parentObject.objectDragged( deltax, deltay )


	def  doubleClick( self, shapeName ):
		self.parentObject.popupEditor()


	def getShapeDescriptor( self, shapeName ):
		return self.parentObject.getProperty( OB_SHAPEDESCRIPTORLIST ).getDescriptor( shapeName )


	def addHandlers( self, canvasObject, aName ):
		canvasObject.connect('event', self.rect_event, aName )



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
			deltax = event.x - oldx
			deltay = event.y - oldy
			self.lastmousex = event.x
			self.lastmousey = event.y
			self.mouseDrag( shapeName, deltax, deltay, oldx, oldy )
			
		elif event.type == gtk.gdk._2BUTTON_PRESS:
			if event.button == 1:
				self.doubleClick( shapeName )

