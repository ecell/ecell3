#!/usr/bin/env python

import gtk
import gnome
import gnome.canvas
import gtk.gdk
import pango

def rect_event( *args ):
	event = args[1]
	item = args[0]
	myobj = args[2]

	if event.type == gtk.gdk.BUTTON_PRESS:
		myobj.lastmousex = event.x
		myobj.lastmousey = event.y
		myobj.buttonpressed = True

	elif event.type == gtk.gdk.BUTTON_RELEASE:
		#item.set_data('buttonpressed', False)
		myobj.buttonpressed = False
		pass

	elif event.type == gtk.gdk.MOTION_NOTIFY:
		if not myobj.buttonpressed:
			return
		oldx = myobj.lastmousex
		oldy = myobj.lastmousey
		myobj.lastmousex = event.x
		myobj.lastmousey = event.y
		deltax = event.x - oldx
		deltay = event.y - oldy
		item.move(deltax,deltay)
		
	elif event.type == gtk.gdk._2BUTTON_PRESS:

#		myobj.buttonpressed = False
#		item.destroy()
#		pass
#		gdkc = item.get_property("fill_color_gdk")

#		newgdkc = gdkc.copy()
#		newgdkc.red= gdkc.red+10000
#		newgdkc.green= gdkc.green+20000
#		newgdkc.blue= gdkc.blue+40000
		
#		item.set_property("fill_color_gdk", newgdkc)
#		x2 = item.get_property("x2" )
#		y2 = item.get_property("y2")
#		x2+=3
#		y2+=5
#		item.set_property("x2", x2)
#		item.set_property("y2", y2)


		myobj.c_size += 0.5
#		pangosize = 1024 * myobj.c_size
#		myobj.pgfd.set_size( pangosize )
		canv.set_pixels_per_unit(myobj.c_size)
#		print myobj.pgfd.get_size()
#		print myobj.pgfd.to_string()
#		myobj.text.set_property('font-desc', myobj.pgfd )
	elif event.type == gtk.gdk.ENTER_NOTIFY:
		print "enter window", event.x, event.y

def canv_event(  *args ):
	pass
	event = args[1]
	item = args[0]
	if event.type == gtk.gdk.ENTER_NOTIFY:
		print "enter canvas ", event.x, event.y

		



def setup_canvas( myobj ):
	canv.set_scroll_region(0,0,200,200)
	canv.set_pixels_per_unit(2)
	rt = canv.root()

	line = rt.add(gnome.canvas.CanvasLine,points=[20,30,80,90], first_arrowhead = gtk.TRUE, last_arrowhead = gtk.TRUE,width_units=2, fill_color="blue", arrow_shape_a=10, arrow_shape_b=10, arrow_shape_c=20, spline_steps = 5 )
	rect = rt.add(gnome.canvas.CanvasRect, x1=40, y1=40, x2=80, y2=80, outline_color="brown", fill_color = "red")
	myobj.text = rt.add( gnome.canvas.CanvasText, x=30, y=30, text="anyad!", fill_color="red", scale = 4, scale_set= True )
	myobj.pgfd = myobj.text.get_property("font-desc").copy()
	#rect.move(100,100)0,200)

	rect.connect('event', rect_event, myobj)
	line.connect('event', rect_event, myobj)
	#rt.connect('event', rect_event)



def window_deleted(*args):
	gtk.mainquit()

class myclass:
	def __init__( self ):
		self.buttonpressed = False
		self.lastmousex = 0
		self.lastmousey = 0
		self.c_size= 0.5
		self.pgfd = None

w= gtk.Window()

w.set_title("Test gnome canvas")
w.connect("delete-event", window_deleted)
v=gtk.VBox()
s=gtk.ScrolledWindow()
myobj = myclass()
canv = gnome.canvas.Canvas()

setup_canvas(myobj)
s.add(canv)
v.pack_start(s)

w.add(v)
w.set_default_size(100,100)
w.show_all()
style = canv.get_style()
style2 = style.copy()
wind = canv.window
colormap=wind.get_colormap()
color = colormap.alloc_color("white")
color.red =1000
style2.bg[0] = color
canv.set_style(style2)
canv.connect('leave-notify-event', canv_event )
canv.connect('enter-notify-event', canv_event )
canv.window.set_cursor( gtk.gdk.Cursor(gtk.gdk.FLEUR ) )
gtk.mainloop()
