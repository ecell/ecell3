#!/usr/bin/env python
from Utils import *
import gtk

import os
import os.path

from ModelEditor import *
from ViewComponent import *
from Constants import *
from ShapePropertyComponent import *
from LinePropertyEditor import *


class ConnectionObjectEditorWindow(ViewComponent):
	
	#######################
	#    GENERAL CASES    #
	#######################

	def __init__( self, aParentWindow, pointOfAttach ):
		
		
		self.theModelEditor = aParentWindow.theModelEditor
		
		ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'ConnectionObjectEditorWindow.glade' )


		#Add Handlers

		self.addHandlers({ 
			'on_BrowseButton_clicked' : self.__FullIDBrowse_displayed,\
			'on_conn_notebook_switch_page' : self.__select_page,\
		        'on_EditButton_clicked' : self.__change_var_reference
			})

		
                # initate Editors
		#self.theLineProperty = LinePropertyEditor(self.theParentWindow, self
#['LinePropertyFrame'] ) 


        def close( self ):
		"""
		closes subcomponenets
		"""
		#close thaw's part
		ViewComponent.close(self)

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

	def __change_var_reference( self, *args ):
		self.getNewVarRefValue()
		

	#########################################
	#    Private methods			#
	#########################################

	def getNewVarRefValue(self):

            	aVarName=(ViewComponent.getWidget(self,'ent_varname')).get_text()
		aVarFullId=(ViewComponent.getWidget(self,'ent_id')).get_text()
		aVarCoef=(ViewComponent.getWidget(self,'ent_coef')).get_text()

		
		
		self.setNewVarRefValue(aVarName,aVarFullId,aVarCoef)

	def setNewVarRefValue(self,name,id,coef):
		print 'The new Var Name: ' + name
		print 'The new Full Id: ' + id
		print 'The coeff: ' + coef
		
		

		

	


	






