
import gtk
import gtk.glade
import os
import gtk.gdk

from MainWindow import *


class  AboutSessionMonitor:
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aMainWindow ):
		"""
		sets up a modal dialogwindow displaying 
		the AboutSessionMonitor window
             
		""" 
		self.theMainWindow = aMainWindow	
		
		filename = os.environ["OSOGOPATH"] + os.sep + "AboutSessionMonitor.glade"
		widgets=gtk.glade.XML(filename,"attachment_box")
		att_box = widgets.get_widget("attachment_box")

		# Create the Dialog
		self.win = gtk.Dialog('AboutSessionMonitor' , None)
		self.win.connect("destroy",self.destroy)

		# Sets size and position
		self.win.set_border_width(2)
		self.win.set_default_size(300,75)
		self.win.set_position(gtk.WIN_POS_MOUSE)

		# appends ok button
		ok_button = gtk.Button("  OK  ")
		self.win.action_area.pack_start(ok_button,gtk.FALSE,gtk.FALSE,)
		ok_button.set_flags(gtk.CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.show()
		ok_button.connect("clicked",self.destroy)
		self.win.vbox.pack_start( att_box )
		# Sets title
		self.win.set_title('About Session Monitor')
		iconPixbuf = None
		try:
			iconPixbuf = gtk.gdk.pixbuf_new_from_file(os.environ['OSOGOPATH'] + os.sep + "ecell.png")
		except:
			pass
		self.win.set_icon(iconPixbuf)
		widgets.get_widget("label1").set_markup("<b>E-Cell Session Monitor Version " + os.environ["VERSION"] + "</b>")
		self.win.show_all()
		
		self.theMainWindow.toggleAboutSessionMonitor(True,self)
		

		
	
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		
		self.win.destroy()

		self.theMainWindow.toggleAboutSessionMonitor(False,None)
		
