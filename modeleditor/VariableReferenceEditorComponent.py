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
from FullIDBrowserWindow import *
from LayoutCommand import *
from EntityCommand import *


class  VariableReferenceEditorComponent(ViewComponent):
    
    #######################
    #    GENERAL CASES    #
    #######################

    def __init__( self, aParentWindow, pointOfAttach, aLayout,connObj ):
        
        
        self.theModelEditor = aParentWindow.theModelEditor
        self.theLayout =aLayout
        self.theConnObj =connObj
        ViewComponent.__init__( self, pointOfAttach, 'attachment_box', 'VariableReferenceEditorComponent.glade' )


        #Add Handlers

        self.addHandlers({ 
            'on_BrowseButton_clicked' : self.__FullIDBrowse_displayed,\
            'on_conn_notebook_switch_page' : self.__select_page,\
                'on_EditButton_clicked' : self.__change_allVar_reference,\
            'on_check_accFlag_toggled' : self.__change_flag_reference,\
            'on_ent_ProName_focus_out_event':self.__changeProcess_displayed,\
            'on_ent_ProName_activate':self.__changeProcess_displayed,\
            'on_ent_coef_editing_done' : self.__change_coef_reference,\
            'on_ent_varid_focus_out_event' : self.__change_varname_reference,\
            'on_ent_varid_activate' : self.__change_varname_reference,\
            'on_varAbs_toggled': self.__change_varAbs_reference,\
            'on_ent_conname_editing_done' : self.__change_conname_reference
                    
            })
        
                # initate Editors
        self.theLineProperty = LinePropertyComponent(self,aParentWindow, connObj ,self
['LinePropertyFrame'] ) 
        
        self.update()


    #########################################
    #    Private methods/Signal Handlers    #
    #########################################


    def __FullIDBrowse_displayed( self, *args ):
        aFullIDBrowserWindow = FullIDBrowserWindow( self, convertSysPathToSysID(self.proFullID.split(':')[1] ) )
        aVariableRef = aFullIDBrowserWindow.return_result()
        
        if aVariableRef == None:
            return
        if getFullIDType( aVariableRef ) != ME_VARIABLE_TYPE:
            return
        if not self.showAbs:
            if isAbsoluteReference( aVariableRef ):
                aVariableRef = getRelativeReference( self.proFullID,  aVariableRef )
    
        ViewComponent.getWidget(self,'ent_varid').set_text(aVariableRef) 
        

    def __select_page( self, *args ):
        pass
    
    def __changeProcess_displayed( self, *args ):
        pass

    def __change_varname_reference( self, *args ):
        pass
    def __change_varAbs_reference(self, *args ):
        if ViewComponent.getWidget(self,'varAbs').get_active() :
            self.showAbs = True
        else:
            self.showAbs = False
        self.changeVarFullID()

    def __change_coef_reference( self, *args ):
        aVarCoef=(ViewComponent.getWidget(self,'ent_coef')).get_text()
        
        self.changeCoef(aVarCoef)

    def __change_conname_reference( self, *args ):
        avarreffName=(ViewComponent.getWidget(self,'ent_conname')).get_text()
        self.changeConnName(avarreffName)

    def __change_flag_reference (self, *args ):
        pass

    def __change_allVar_reference( self, *args ):
        self.getNewVarRefValue()



    

    #########################################
    #    Private methods            #
    #########################################
    def updateVarRef(self):
        ViewComponent.getWidget(self,'ent_conname').set_sensitive(gtk.TRUE)
        ViewComponent.getWidget(self,'ent_coef').set_sensitive(gtk.TRUE)
        ViewComponent.getWidget(self,'varAbs').set_sensitive(gtk.TRUE)
        ViewComponent.getWidget(self,'ent_ProName').set_text(self.proFullID )
        ViewComponent.getWidget(self,'ent_conname').set_text(self.varreffName)
        ViewComponent.getWidget(self,'ent_varid').set_text( self.varFullID)
        ViewComponent.getWidget(self,'ent_coef').set_text( str(self.acoef) )
        if self.showAbs :
            ViewComponent.getWidget(self,'varAbs').set_active(gtk.TRUE)
        elif not self.showAbs :
            ViewComponent.getWidget(self,'varAbs').set_active(gtk.FALSE)
        
    def setDisplayedVarRef(self, aLayout,connObj):
        self.theLayout = aLayout
        self.theConnObj = connObj
        self.update()

    def update( self ):
        if self.theConnObj ==  None  : 
            self.clearVarRef()
            self.theLineProperty.setDisplayedLineProperty(self.theConnObj)
            return 
        else:
            
            existObjectList = self.theLayout.getObjectList()
            self.theConnObjID = self.theConnObj.getID()
            self.varreffName = self.theConnObj.getProperty(CO_NAME)
            self.acoef = self.theConnObj.getProperty(CO_COEF)

            
            if self.theConnObjID not in existObjectList : 
                self.clearVarRef()
                self.theConnObj =  None
                self.theLineProperty.setDisplayedLineProperty(self.theConnObj)
                return
            

            proID = self.theConnObj.getProperty(CO_PROCESS_ATTACHED)
            if proID not in existObjectList:
                self.clearVarRef()
                self.theConnObj =  None
                self.theLineProperty.setDisplayedLineProperty(self.theConnObj)
                return
                
            else:   
            
                varID = self.theConnObj.getProperty(CO_VARIABLE_ATTACHED)
                if varID != None and varID not in existObjectList:
                    self.clearVarRef()
                    self.theConnObj =  None
                    self.theLineProperty.setDisplayedLineProperty(self.theConnObj)
                    return
                else:   
                    self.proObject = self.theLayout.getObject( proID )
                    self.proFullID = self.proObject.getProperty(OB_FULLID)
                    if not self.theModelEditor.getModel().isEntityExist(self.proFullID):
                        varObj = self.theLayout.getObject( varID )
                        self.varFullID = varObj.getProperty(OB_FULLID)
                    else:
                        aProFullPN = createFullPN ( self.proFullID, MS_PROCESS_VARREFLIST )
                        aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aProFullPN) )
                        for aVarref in aVarrefList:
                            if aVarref[ME_VARREF_NAME] == self.varreffName:
                                self.varFullID = aVarref[ME_VARREF_FULLID]
        
                        
                    self.checkVarAbs()
                    self.theLineProperty.setDisplayedLineProperty(self.theConnObj)
    
                    
            
        
    def clearVarRef(self):
        self.proFullID = ''
        self.varreffName=''
        self.varFullID=''
        self.acoef=''
        self.updateVarRef()
        ViewComponent.getWidget(self,'ent_conname').set_sensitive(gtk.FALSE)
        ViewComponent.getWidget(self,'ent_coef').set_sensitive(gtk.FALSE)
        ViewComponent.getWidget(self,'varAbs').set_sensitive(gtk.FALSE)
    
    def checkVarAbs(self):
        if self.varFullID == None:
            if self.showAbs:
                self.varFullID  = ':/___NOTHING'
            else:
                self.varFullID  = ':.:___NOTHING'
        elif isAbsoluteReference( self.varFullID ):
            self.showAbs = True
        else:
            self.showAbs = False
        self.updateVarRef()

    def changeVarFullID(self):      
        
        if self.showAbs:
            if not isAbsoluteReference( self.varFullID ):
                self.varFullID = getAbsoluteReference( self.proFullID , self.varFullID )
                aProFullPN = createFullPN ( self.proFullID, MS_PROCESS_VARREFLIST )
                aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aProFullPN) )
                for aVarref in aVarrefList:
                    if aVarref[ME_VARREF_NAME] ==self.varreffName:
                        aVarref[ME_VARREF_FULLID] = self.varFullID
                        aCommand = ChangeEntityProperty( self.theModelEditor, aProFullPN, aVarrefList )
                self.theLayout.passCommand( [ aCommand ] )

        elif not self.showAbs:
            if isAbsoluteReference( self.varFullID ):
                self.varFullID = getRelativeReference( self.proFullID,  self.varFullID)
                aProFullPN = createFullPN ( self.proFullID, MS_PROCESS_VARREFLIST )
                aVarrefList = copyValue( self.theModelEditor.getModel().getEntityProperty( aProFullPN) )
                for aVarref in aVarrefList:
                    if aVarref[ME_VARREF_NAME] ==self.varreffName:
                        aVarref[ME_VARREF_FULLID] = self.varFullID
                        aCommand = ChangeEntityProperty( self.theModelEditor, aProFullPN,aVarrefList )
                self.theLayout.passCommand( [ aCommand ] )

        
        self.updateVarRef()

    def changeConnName(self,newName):
        
        aVarReffList =copyValue(self.theModelEditor.theModelStore.getEntityProperty(self.proFullID+':' +MS_PROCESS_VARREFLIST))
        existReffname =[]
        for aVarRef in aVarReffList: 
            existReffname += [aVarRef[MS_VARREF_NAME]]
        if newName not in existReffname:    
            for aVarRef in aVarReffList: 
                if self.varreffName == aVarRef[MS_VARREF_NAME] :
                    aVarRef[MS_VARREF_NAME] = newName
                    aCommand = ChangeEntityProperty( self.theModelEditor, self.proFullID+':' +MS_PROCESS_VARREFLIST, aVarReffList )
                    self.theLayout.passCommand( [ aCommand ] )
        
        self.checkVarAbs()

        

    def changeCoef(self,newCoef):
        try :
            newCoef = int( newCoef)
        except:
            self.checkVarAbs()
            return None
        
        aCommand = None
        aVarReffList =copyValue(self.theModelEditor.theModelStore.getEntityProperty(self.proFullID+':' +MS_PROCESS_VARREFLIST))
        for aVarRef in aVarReffList: 
            if self.varreffName == aVarRef[MS_VARREF_NAME]:
                if aVarRef[MS_VARREF_COEF]!=newCoef:
                    aVarRef[MS_VARREF_COEF] = newCoef
                    aCommand = ChangeEntityProperty( self.theModelEditor, self.proFullID+':' +MS_PROCESS_VARREFLIST, aVarReffList )
        if aCommand !=None:
            self.theLayout.passCommand( [ aCommand ] )

        self.checkVarAbs()
