#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2008 Keio University
#       Copyright (C) 2005-2008 The Molecular Sciences Institute
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
#
# Written by Hiep
# 
#

import os
import os.path
import gtk 

import ecell.ui.model_editor.Config as config
from ecell.ui.model_editor.ModelEditor import * 
from ecell.ui.model_editor.Window import *
from ecell.ui.model_editor.ConfirmWindow import *
from ecell.ui.model_editor.ViewComponent import *

OK_MODE = 0
OKCANCEL_MODE = 1

# Constans for result
OK_PRESSED = 0
CANCEL_PRESSED = -1


class AutosaveWindow:
       

        # ==========================================================================
        def __init__(self, aModelEditor, aDuration):
               
            
            self.theModelEditor = aModelEditor
            # Sets the return number

            self.___num = CANCEL_PRESSED
            self.__off = False
  
            self.win = gtk.Dialog('AutosaveWindow' , None)
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
            ok_button.connect("clicked",self.okButtonClicked)

            # appends cancel button
            cancel_button = gtk.Button(" Cancel ")
            self.win.action_area.pack_start(cancel_button,False,False)
            cancel_button.show()
            cancel_button.connect("clicked",self.cancelButtonClicked)

            # Sets title
            self.win.set_title('Preferences')
        
           
            self.ViewComponentObject = ViewComponent( self.win.vbox, 'attachment_box',  'AutosaveWindow.glade' )
            self.ViewComponentObject['duration']
          
            aPixbuf16 = gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.GLADEFILE_PATH, "modeleditor.png") )
            aPixbuf32 = gtk.gdk.pixbuf_new_from_file(
                os.path.join( config.GLADEFILE_PATH, "modeleditor32.png") )
            self.win.set_icon_list(aPixbuf16, aPixbuf32)
        
            self.win.show_all()
            self.__setAutosaveDuration ( aDuration )
            self.ViewComponentObject.addHandlers({ "on_duration_toggled" : self.__buttonChosen,                "on_operations_toggled" : self.__buttonChosen,
"on_turn_off_toggled" : self.__buttonChosen })
            gtk.main()
        
            
        def __setAutosaveDuration ( self, aDuration ):
            if aDuration[0]>0:
                self.ViewComponentObject['set_duration'].set_active(True)
                self.ViewComponentObject['duration'].set_text(str(aDuration[0]))
                self.ViewComponentObject['operations'].set_sensitive(False)
                self.ViewComponentObject['duration'].set_sensitive(True)
            elif aDuration[1]>0:
                self.ViewComponentObject['set_operations'].set_active(True)
                self.ViewComponentObject['operations'].set_text(str(aDuration[1]))
                self.ViewComponentObject['operations'].set_sensitive(True)
                self.ViewComponentObject['duration'].set_sensitive(False) 
            else:
                self.ViewComponentObject['turn_off'].set_active( True)
                self.ViewComponentObject['duration'].set_sensitive(False)
                self.ViewComponentObject['operations'].set_sensitive(False)

        def cancelButtonClicked( self, *arg ):
            """
            If Cancel button clicked or the return pressed, this method is called.
            """
            # set the return number
            self.___num = None
            self.destroy()

   
        def destroy( self, *arg ):
            """destroy dialog
            """
        
            self.win.hide()
            gtk.main_quit()

        def __saveAutosaveDuration ( self ):
              
            aDuration = [1000,5]
            if self.__off == True:
                aDuration = [0,0]
                          
            else:
                #radiobutton(set_duration) is selected              
                if self.ViewComponentObject['set_duration'].get_active() == True:      
                    try:
                
                        num = self.ViewComponentObject['duration'].get_text()
                        aDuration[0] = int(num)
                        if aDuration[0]<1:
                            a=1/0
                        aDuration[1] = 0
                    except:      
              
                
                        self.theModelEditor.openConfirmWindow( "Please enter valid positive integer for time duration", "Invalid number format", 0)
                        return None
                else:
                    try:
                
                               
                        num = self.ViewComponentObject['operations'].get_text()
                        aDuration[1] = int(num)
                    
                        if aDuration[1]<0:
                            a=1/0  
                        aDuration[0] = 0
                    except:
                        self.theModelEditor.openConfirmWindow( "Please enter valid positive integer for number of operations", "Invalid number format" , 0)
                        return None    
                        

            return aDuration
                

        def okButtonClicked ( self, *arg ):
            aDuration = self.__saveAutosaveDuration()
            if aDuration == None:
                return
            self.___num = aDuration
            self.destroy()
            
        
        def return_result( self ):
            """Returns result
            """
            return self.___num

        def __buttonChosen(self, *args):
            aName = args[0].get_name()
            if aName == "turn_off":
                self.__off = True
                self.ViewComponentObject['duration'].set_sensitive(False)
                self.ViewComponentObject['operations'].set_sensitive(False)
                
            else:    
                
                self.__off = False
                if aName == "set_duration":
                    self.ViewComponentObject['duration'].set_sensitive(True)
                    self.ViewComponentObject['operations'].set_sensitive(False)
                else:
                    self.ViewComponentObject['duration'].set_sensitive(False)
                    self.ViewComponentObject['operations'].set_sensitive(True)
            
                


          
                
              
              
              
        

        
