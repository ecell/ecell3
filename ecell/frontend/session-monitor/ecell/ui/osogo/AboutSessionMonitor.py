#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2016 Keio University
#       Copyright (C) 2008-2016 RIKEN
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

import gtk
import os
import gtk.gdk

import ecell.ui.osogo.config as config
from ecell.ui.osogo.MainWindow import *
import ecell.ui.osogo.glade_compat as glade

class AboutSessionMonitor:
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aMainWindow ):
		"""
		sets up a modal dialogwindow displaying 
		the AboutSessionMonitor window
             
		""" 
		self.theMainWindow = aMainWindow	
		
		filename = os.path.join( config.GLADEFILE_PATH, "AboutSessionMonitor.glade" )
		widgets = glade.XML(filename,"attachment_box")
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
		self.win.action_area.pack_start(ok_button,False,False,)
		ok_button.set_flags(gtk.CAN_DEFAULT)
		ok_button.grab_default()
		ok_button.show()
		ok_button.connect("clicked",self.destroy)
		self.win.vbox.pack_start( att_box )
		# Sets title
		self.win.set_title('About Session Monitor')
		aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, 'ecell.png') )
		aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
            os.path.join( config.GLADEFILE_PATH, 'ecell32.png') )
		self.win.set_icon_list(aPixbuf16, aPixbuf32)
		widgets.get_widget("label1").set_markup("<b>E-Cell Session Monitor Version " + config.version + "</b>")
		self.win.show_all()
		
		self.theMainWindow.toggleAboutSessionMonitor(True,self)
		

		
	
	def destroy( self, *arg ):
		"""destroy dialog
		"""
		
		self.win.destroy()

		self.theMainWindow.toggleAboutSessionMonitor(False,None)
		
