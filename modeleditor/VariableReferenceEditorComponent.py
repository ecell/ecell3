#!/usr/bin/env python


#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#		This file is part of E-CELL Model Editor package
#
#				Copyright (C) 1996-2003 Keio University
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
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Sylvia Tarigan' at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#

from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from ShapePropertyComponent import *
from LinePropertyComponent import *


class  VariableReferenceEditorComponent(ViewComponent):
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aParentWindow, pointOfAttach ):
		
		
		self.theModelEditor = aParentWindow.theModelEditor
		
		ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'VariableReferenceEditorComponent.glade' )


		#Add Handlers

		self.addHandlers({ 
			'on_BrowseButton_clicked' : self.__FullIDBrowse_displayed,\
			'on_conn_notebook_switch_page' : self.__select_page,\
		        'on_EditButton_clicked' : self.__change_allVar_reference,\
			'on_check_accFlag_toggled' : self.__change_flag_reference,\
			'on_ent_ProName_focus_out_event':self.__changeProcess_displayed,\
			'on_ent_ProName_activate':self.__changeProcess_displayed,\
			'on_ent_coef_focus_out_event' : self.__change_coef_reference,\
			'on_ent_coef_activate' : self.__change_coef_reference,\
			'on_ent_varname_focus_out_event' : self.__change_varname_reference,\
			'on_ent_varname_activate' : self.__change_varname_reference,\
			'on_ent_id_changed' : self.__change_id_reference,\
			'on_ent_id_activate' : self.__change_id_reference
					
			})

		
                # initate Editors
		self.theLineProperty = LinePropertyComponent(self, self
['LinePropertyFrame'] ) 


       


	#########################################
	#    Private methods/Signal Handlers    #
	#########################################


	def __FullIDBrowse_displayed( self, *args ):
		self.FullIDBrowser = FullIDBrowserWindow(self, ME_VARIABLE_TYPE)
		self.NewFullID = self.FullIDBrowser.return_result()
		if self.NewFullID != None:
			(ViewComponent.getWidget(self,'ent_id')).set_text(self.NewFullID) 
		else:
			pass

	def __select_page( self, *args ):
		pass
	
	def __changeProcess_displayed( self, *args ):
		pass

	def __change_varname_reference( self, *args ):
		pass

	def __change_coef_reference( self, *args ):
		pass


	def __change_id_reference( self, *args ):
		pass

	def __change_flag_reference (self, *args ):
		pass

	def __change_allVar_reference( self, *args ):
		self.getNewVarRefValue()



	

	#########################################
	#    Private methods			#
	#########################################

	def getNewVarRefValue(self):

            	aVarName=(ViewComponent.getWidget(self,'ent_varname')).get_text()
		aVarFullId=(ViewComponent.getWidget(self,'ent_id')).get_text()
		aVarCoef=(ViewComponent.getWidget(self,'ent_coef')).get_text()
		aAccel=(ViewComponent.getWidget(self,'check_accFlag')).get_active()

		
		
		self.setNewVarRefValue(aVarName,aVarFullId,aVarCoef,aAccel)

	

	def setNewVarRefValue(self,name,id,coef,accel):
		print 'The new Var Name: ' + name
		print 'The new Full Id: ' + id
		print 'The coeff: ' + coef
		print 'The Accel:' + str(accel)
		
		

		

	


	






