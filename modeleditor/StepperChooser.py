#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of E-CELL Model Editor package
#
#               Copyright (C) 1996-2003 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-CELL is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
#
# E-CELL is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public
# License along with E-CELL -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
#
#END_HEADER
#
#'Design: Gabor Bereczki <gabor@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Gabor Bereczki' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


from Utils import *
import gtk

import os
import os.path
import gobject

from ModelEditor import *
from ViewComponent import *

class StepperChooser(ViewComponent):

    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach ):
        self.theParentWindow = aParentWindow
        # call superclass
        ViewComponent.__init__( self,   pointOfAttach, 'attachment_box' )
    
        # add handler
        self.addHandlers( {'on_stepper_choser_entry_changed' : self.__entry_changed } )

        # set up textbuffer
        self.theTextBuffer = gtk.TextBuffer()
        self['stepper_choser_textview'].set_buffer( self.theTextBuffer )
        
        # get stepper list
        self.theModelEditor = self.theParentWindow.theModelEditor
        self.theStepperList = self.theModelEditor.getModel().getStepperList()
        self['combo1'].set_popdown_strings( self.theStepperList )

        # init popdown strings
        if self.theParentWindow.thePropertyValue !='':
            self['stepper_choser_entry'].set_text( self.theParentWindow.thePropertyValue )
        elif len(self.theStepperList)> 0:
            self['stepper_choser_entry'].set_text(self.theStepperList[0] )
        else:
            self['stepper_choser_entry'].set_text('')
        self.__entry_changed( None )


    def getValue( self ):
        return self['stepper_choser_entry'].get_text()



    def __entry_changed( self, *args):
        theStepper = self['stepper_choser_entry'].get_text()
        if theStepper == '':
            theText = ''
        else:
            if theStepper in self.theModelEditor.getModel().getStepperList():
                theText = self.theModelEditor.getModel().getStepperInfo( theStepper )
            else:
                theText = "#DELETED"
        self.theTextBuffer.set_text( theText )

