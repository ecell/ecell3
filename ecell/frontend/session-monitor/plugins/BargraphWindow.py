#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
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

import os
import re
import operator
from numpy import *
import gtk 
import gtk.gdk

from ecell.ecssupport import *

import ecell.ui.osogo.config as config
from ecell.ui.osogo.Plot import *
from ecell.ui.osogo.OsogoPluginWindow import *

class BargraphWindow( OsogoPluginWindow ):
	def __init__( self, dirname, data, pluginmanager, rootWidget = None ):
		#initializa variables:
		#initiates window
		OsogoPluginWindow.__init__(self, dirname, data, pluginmanager,\
					   rootWidget )
		self.theSession = pluginmanager.theSession
		self.BAR_WIDTH = 150
		self.BAR_HEIGTH = 20

	def openWindow( self ):
		OsogoPluginWindow.openWindow(self)
		self.handlebox = self.getWidget('handlebox1')
		self.drawingarea = self.getWidget('drawingarea1')
		self.numberlabel = self.getWidget('label4')
		self.textlabel = self.getWidget('label5')
		self.arrow = self.getWidget('image1')
		self.button = self.getWidget('togglebutton1')
		self.drawingframe = self.getWidget('frame8')
		self.buttonstate = 1
		self.button.set_active(self.buttonstate)

		self.ColorMap = self.handlebox.get_colormap()
		#sets colorcodes, codetable
		#codetable: list, range upper boundaries in ascendingorder, GDKColor codes
		self.colored_ranges = [
			    [-1,'black',None], #overload
			    [0,'grey',None], #default background
			    [10,'blue',None],
			    [100,'yellow',None],
			    [1000,'red',None]]
			    
		for i in range(len(self.colored_ranges)):
		        aRootWindow = self.getParent()
		        root = aRootWindow[aRootWindow.__class__.__name__]
		        gc = root.window.new_gc()
		        gc.set_foreground(\
		        self.ColorMap.alloc_color(self.colored_ranges[i][1]))
		        self.colored_ranges[i][2] = gc

		#lastvalue zero
		self.lastvalue = 0
		self.lastscale = 0
		self.lastposition = 0
		#paint pixbuf according to lastvalue - draw everything to default
		self.pm = gtk.gdk.Pixmap(root.window,self.BAR_WIDTH,self.BAR_HEIGTH,-1)
		
		self.pm.draw_rectangle(self.colored_ranges[1][2],True,0,0,
			    self.BAR_WIDTH,self.BAR_HEIGTH)

		self.addHandlers({\
		    'drawingarea1_expose_event':self.expose,
		    'togglebutton1_toggled':self.press})

		self.drawingarea.queue_draw_area(0,0,self.BAR_WIDTH, self.BAR_HEIGTH)			
		#calls update
		self.thePluginManager.appendInstance(self)
		self.ccFullPN = self.getFullPNList()[0]
		nameFullPN = createFullPNString(self.ccFullPN)
		self.textlabel.set_text(nameFullPN)
                self.setIconList(
			os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
			os.path.join( config.GLADEFILE_PATH, "ecell32.png")
			)
		self.update()

	def update(self):
	    #getlatest data
	    #currentvalue = self.theFullPN
	    self.current_value = self.getlatestdata()
	    self.numberlabel.set_text(str(self.current_value))
	    #calculate scale
	    self.current_scale = self.get_scale(self.current_value)
	    if self.current_scale == 0:
	    #paint the whole area black, lastvalue, lastposition, lastscale = 0
		self.pm.draw_rectangle(self.colored_ranges[0][2],True,0,0,
			    self.BAR_WIDTH,self.BAR_HEIGTH)
		self.drawingarea.queue_draw_area(0,0,self.BAR_WIDTH,self.BAR_HEIGTH)
		self.lastvalue = 0
		self.lastposition = 0
		self.lastscale = 0
	    else:
		#calculate direction
		difference = self.current_value-self.lastvalue
		if difference>0:
		    icon_id = gtk.STOCK_GO_UP
		elif difference<0:
		    icon_id = gtk.STOCK_GO_DOWN
		else:
		    icon_id = gtk.STOCK_REMOVE
		self.arrow.set_from_stock(icon_id,4)  
		self.current_position = int(self.convert_value(self.current_value))
		#if scalechange, drawall
		if self.lastscale != self.current_scale:
		    #draw painted area and draw grey area
		    self.pm.draw_rectangle(self.colored_ranges[self.current_scale][2],
			True,0,0,self.current_position,self.BAR_HEIGTH)
		    self.pm.draw_rectangle(self.colored_ranges[1][2],True,
			self.current_position,0,self.BAR_WIDTH-self.current_position,
			self.BAR_HEIGTH)
		    self.drawingarea.queue_draw_area(0,0,self.BAR_WIDTH,self.BAR_HEIGTH)
		else:	
		#if not scalechange 
		    if difference<0:
			brush = self.colored_ranges[1][2]
			x0 = self.current_position
			width = self.lastposition-self.current_position
		    #if difference is negativ, paint with grey,
		    elif difference>0:
			brush = self.colored_ranges[self.current_scale][2]
			x0 = self.lastposition
			width = self.current_position-self.lastposition
		    #if difference ispositiv , paint with proper color
		    else:
			return True #do not draw if no changes
		    self.pm.draw_rectangle(brush,True,x0,0,width,self.BAR_HEIGTH)
		    self.drawingarea.queue_draw_area(x0,0,width,self.BAR_HEIGTH) 
		self.lastvalue = self.current_value
		self.lastscale = self.current_scale
		self.lastposition = self.current_position
	    return True
	
	def convert_value(self,aValue):
	    if self.current_scale>1:
		return round((aValue-self.colored_ranges[self.current_scale-1][0])/\
		    self.colored_ranges[self.current_scale][0]*self.BAR_WIDTH)
	    else:
		return 0
		
	def expose(self, obj, event):
	    alloc_rect = self.drawingarea.get_allocation()
	    new_width = alloc_rect[2]
	    new_heigth = alloc_rect[3]
	    if new_width != self.BAR_WIDTH or new_heigth != self.BAR_HEIGTH:
		#needs resize
		self.BAR_WIDTH = new_width
		self.BAR_HEIGTH = new_heigth
		aRootWindow = self.getParent()
		root = aRootWindow[aRootWindow.__class__.__name__]
		self.pm = gtk.gdk.Pixmap(root.window,self.BAR_WIDTH,self.BAR_HEIGTH,-1)
		
		self.lastscale = -2
		self.update()
	    obj.window.draw_drawable(self.pm.new_gc(),self.pm,event.area[0],event.area[1],
				    event.area[0],event.area[1],event.area[2],event.area[3])
	
	def press(self, obj):
	    self.button.set_active(self.buttonstate)

	def getlatestdata(self):
	    return self.theSession.getEntityProperty( createFullPNString( self.ccFullPN ) )
	    
	def get_scale(self,value): 
	    no_scales = len(self.colored_ranges)
	    if value<self.colored_ranges[1][0] or\
		value>self.colored_ranges[no_scales-1][0]:
		#the whole bar should be painted black
		return 0
	    else:
		for i in range(2,no_scales):
		    if self.colored_ranges[i][0] >= value:
			return i

		
