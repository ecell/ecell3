#!/usr/bin/env python

import gtk
import gnome
import gnome.canvas


def rect_event( *args ):
	event = args[1]
	item = args[0]
	print item, event.type
	if event.type == gtk.gdk.BUTTON_PRESS:
		item.set_data('lastmousex', event.x)
		item.set_data('lastmousey', event.y)
		item.set_data('buttonpressed', True)

	elif event.type == gtk.gdk.BUTTON_RELEASE:
		item.set_data('buttonpressed', False)

	elif event.type == gtk.gdk.MOTION_NOTIFY:
		if not item.get_data('buttonpressed'):
			return
		oldx = item.get_data('lastmousex')
		oldy = item.get_data('lastmousey')
		item.set_data('lastmousex', event.x)
		item.set_data('lastmousey', event.y)
		deltax = event.x - oldx
		deltay = event.y - oldy
		item.move(deltax,deltay)
		#x1 = item.get_property('x1')
		#y1 = item.get_property('y1')
		#print "x1", x1, "y1", y1
		#print "deltax
		
		


def setup_canvas():
	canv.set_scroll_region(0,0,200,200)
	canv.set_pixels_per_unit(1)
	rt = canv.root()

	line = rt.add(gnome.canvas.CanvasLine,points=[20,30,80,90], first_arrowhead = gtk.TRUE, last_arrowhead = gtk.TRUE,width_units=2, fill_color="blue", arrow_shape_a=10, arrow_shape_b=10, arrow_shape_c=20, spline_steps = 5 )
	line.set_data('lastmousex',0)
	line.set_data('lastmousey',0)
	line.set_data('buttonpressed',False)
	rect = rt.add(gnome.canvas.CanvasRect, x1=40, y1=40, x2=80, y2=80, outline_color="brown", fill_color = "red")
	rect.set_data('lastmousex',0)
	rect.set_data('lastmousey',0)
	rect.set_data('buttonpressed',False)
	
	#rect.move(100,100)
	rect.connect('event', rect_event)
	line.connect('event', rect_event)
	rt.connect('event', rect_event)


def window_deleted(*args):
	gtk.mainquit()


w= gtk.Window()
w.set_title("Test gnome canvas")
w.connect("delete-event", window_deleted)
v=gtk.VBox()
s=gtk.ScrolledWindow()

canv = gnome.canvas.Canvas()
setup_canvas()
s.add(canv)
v.pack_start(s)

w.add(v)
w.set_default_size(100,100)
w.show_all()
gtk.mainloop()
