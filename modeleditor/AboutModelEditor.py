
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *



class  AboutModelEditor:
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aModelEditor):
		"""
		sets up a modal dialogwindow displaying 
		the AboutModelEditor window
             
		""" 
		self.theModelEditor = aModelEditor	
		
		# Create the Dialog
		self.win = gtk.Dialog('AboutModelEditor' , None)
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

		# Sets title
		self.win.set_title('About Model Editor')
		
		
		ViewComponent( self.win.vbox, 'attachment_box', 'AboutModelEditor.glade' )
		aPixbuf16 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                                os.sep + "glade" + os.sep + "modeleditor.png")
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file( os.environ['MEPATH'] +
                                os.sep + "glade" + os.sep + "modeleditor32.png")
		self.win.set_icon_list(aPixbuf16, aPixbuf32)
		
		self.win.show_all()
		
		self.theModelEditor.toggleAboutModelEditor(True,self)
		

		
	
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		
		self.win.destroy()

		self.theModelEditor.toggleAboutModelEditor(False,None)
		
