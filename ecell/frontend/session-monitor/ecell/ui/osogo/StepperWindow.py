#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation either
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
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>' at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from os import *
import sys
import gobject
import traceback

from ecell.ecssupport import *

from ecell.ui.osogo.ConfirmWindow import *
from ecell.ui.osogo.OsogoUtil import *
from ecell.ui.osogo.OsogoWindow import *

# Constant value of this class
PROPERTY_INDEX = 0
VALUE_INDEX = 1
GET_INDEX = 2
SET_INDEX = 3

MAX_STRING_NUMBER = 40

import copy

class StepperWindow(OsogoWindow):
    """StepperWindow
    - displayes all stepper property
    - user can change each stepper property
    """

    # ==========================================================================
    def __init__( self, aSession ): 
        """Constructor
        aSession     ---  the reference of session
        aSession  ---  the reference of Session
        """

        # calls superclass's constructor
        OsogoWindow.__init__( self, aSession, 'StepperWindow.glade' )

        self.theSession = aSession



    # ==========================================================================
    def openWindow(self):
        OsogoWindow.openWindow(self)

        # initializes attributes
        self.theSelectedStepperID = None  # selected stepperID (str)
        self.theFirstPropertyIter = {}    # key:iter(TreeIter)  value:None
        self.theSelectedPath = {}         # key:aStepperID(str)  value:Path
        self.thePropertyMap = {}

        aListStore = gtk.ListStore( gobject.TYPE_STRING )
        self['stepper_id_list'].set_model( aListStore )
        column=gtk.TreeViewColumn('Stepper',gtk.CellRendererText(),text=0)
        self['stepper_id_list'].append_column(column)

        aPropertyModel = gtk.ListStore(
            gobject.TYPE_STRING,
            gobject.TYPE_STRING,
            gobject.TYPE_BOOLEAN,
            gobject.TYPE_BOOLEAN )
        self['property_list'].set_model(aPropertyModel)

        column=gtk.TreeViewColumn( 'Property',gtk.CellRendererText(),\
                       text=PROPERTY_INDEX )
        column.set_resizable(True)
        self['property_list'].append_column(column)

        renderer = gtk.CellRendererText()
        renderer.connect( 'edited', self.__updateProperty )
        column=gtk.TreeViewColumn( 'Value', renderer,\
                       text=VALUE_INDEX, editable=SET_INDEX, sensitive=SET_INDEX )
        column.set_resizable(True)
        self['property_list'].append_column(column)

        column=gtk.TreeViewColumn( 'Get',gtk.CellRendererToggle(),\
                       active=GET_INDEX )
        column.set_resizable(True)
        self['property_list'].append_column(column)

        column=gtk.TreeViewColumn( 'Set',gtk.CellRendererToggle(),\
                       active=SET_INDEX )
        column.set_resizable(True)
        self['property_list'].append_column(column)
        

        # adds handlers
        self.addHandlers({ \
                'on_stepper_id_list_select' : self.__selectStepperID,  # StepperID list
                'on_close_button_clicked' : self.deleted,            # close button
            })


        aModel = self['stepper_id_list'].get_model()
        aModel.clear()            

        aFirstIter = None
        #for aValue in self.theStepperIDList: 
        for aValue in self.theSession.getStepperList():
            anIter = aModel.append()
            if aFirstIter == None:
                aFirstIter = anIter
            aModel.set( anIter, 0, aValue )

        self[self.__class__.__name__].show_all()

        self['stepper_id_list'].get_selection().select_iter(aFirstIter)
        self.__selectStepperID(None)

    def close( self ):
        self.theSelectedPath = {}
        OsogoWindow.close(self)


    # ==========================================================================
    def selectStepperID( self, aStepperID ):
        """ selects StepperID on screen and displays its property list 
            if StepperID exists returns True, else returns False
        """
        anIter=self['stepper_id_list'].get_model().get_iter_first()
        while True:
            if anIter == None:
                return False
            aTitle = self['stepper_id_list'].get_model().get_value(anIter, 0 )
            if aTitle == aStepperID:
                aPath = self['stepper_id_list'].get_model().get_path ( anIter )
                self['stepper_id_list'].set_cursor( aPath, None, False )
                break
            anIter=self['stepper_id_list'].get_model().iter_next(anIter)
        self.__selectStepperID(  None )
        return False

    

    # ==========================================================================
    def __selectStepperID( self, *arg ):
        """selects stepper ID
        Return None
        """

        # When window is not created, does nothing.
        if not self.exists():
            return None

        # --------------------------------------------------
        # Creates selected StepperSub 
        # --------------------------------------------------
        iter = self['stepper_id_list'].get_selection().get_selected()[1]
        # aStepperID is selected stepper id
        aStepperID = self['stepper_id_list'].get_model().get_value(iter,0)
        
        # When same StepperID is selected, does nothing
        if self.theSelectedStepperID != None and aStepperID == self.theSelectedStepperID:
            return None
        self.theSelectedStepperID = aStepperID
        self.thePropertyMap = {}
        # aStepperStub is selected StepperStub of selected stepper
        aStepperStub = self.theSession.createStepperStub( aStepperID )

        # updates property list
        aPropertyModel = self['property_list'].get_model()
        aPropertyModel.clear()

        # creats list of ClassName's row
        aList = [ 'ClassName', ]

        # value
        aClassName = aStepperStub.getClassname( )
        aList.append( str(aClassName) )
        self.thePropertyMap[ 'ClassName' ] = str( aClassName )

        # gettable and settable
        aList.append( True )   # gettable is '+'
        aList.append( False )  # settable is '-'

        # sets this list to TreeModel
        iter = aPropertyModel.append()
        for i in range(0, 4):
            aPropertyModel.set_value(iter, i, aList[i])
                        
        self.theFirstPropertyIter[aStepperID] = iter

        # --------------------------------------------------
        # sets all propertys' row other than ClassName
        # --------------------------------------------------
        for aProperty in aStepperStub.getPropertyList():

            # property
            aList = [ aProperty, ]  # first element

            # gettable and settable
            anAttribute = aStepperStub.getPropertyAttributes( aProperty )
            # value
            if anAttribute[GETABLE] == 0:
                continue
            aValue = aStepperStub.getProperty( aProperty )
            self.thePropertyMap[ aProperty ] = aValue

            aValueString = str( aValue )
            # second element
            aList.append( shortenString( aValueString,\
                             MAX_STRING_NUMBER) )  

            aList.append( anAttribute[GETABLE] )  # third element
            aList.append( anAttribute[SETTABLE] ) # forth element

            # sets this list to TreeModel
            iter = aPropertyModel.append()
            #anIterListElement = [iter]
            for i in range(0,4):
                aPropertyModel.set_value(iter, i, aList[i])

        self.update()

        if self.theSelectedPath.has_key(aStepperID):
            aPath = self.theSelectedPath[aStepperID]
            self['property_list'].get_selection().select_path(aPath)
        else:
            aPath = (0,)
            self.theSelectedPath[aStepperID] = aPath
            self['property_list'].get_selection().select_path(aPath)

    # ==========================================================================
    def selectProperty(self, aPropertyName):
        """ selects PropertyName on screen  
            if PropertyName exists returns True, else returns False
        """
        anIter=self['property_list'].get_model().get_iter_first()

        while True:
            if anIter == None:
                return False
            aTitle = self['property_list'].get_model().get_value(anIter, PROPERTY_INDEX )
            if aTitle == aPropertyName:
                aPath = self['property_list'].get_model().get_path ( anIter )
                self['property_list'].set_cursor( aPath, None, False )
                break
            anIter=self['property_list'].get_model().iter_next(anIter)

        return False

    # ==========================================================================
    # ==========================================================================
    def __updateProperty( self, renderer, path, aValue, *kwarg ):
        """updates property
        Return None
        """

        # --------------------------------------------------
        # creates selected StepperSub 
        # --------------------------------------------------
        iter = self['stepper_id_list'].get_selection().get_selected()[1]
        aStepperID = self['stepper_id_list'].get_model().get_value(iter,0)
        aStepperStub = self.theSession.createStepperStub( aStepperID )

        # gets selected property row
        aPropertyModel = self['property_list'].get_model()
        iter = aPropertyModel.get_iter(path)
        # -----------------------------------------------------------
        # get a property name from property list
        # -----------------------------------------------------------
        aPropertyName = aPropertyModel.get_value( iter, PROPERTY_INDEX )

        # converts value type
        anOldValue = aStepperStub.getProperty( aPropertyName )

        # ---------------------------------------------------
        # checks float and int type of inputted value
        # does not check str. ( not needed )
        # ---------------------------------------------------
        # float
        if type(anOldValue) == float:
            try:
                aValue = float(aValue)
            except:
                # displays confirm window
                anErrorMessage = "Input float value."
                aDialog = ConfirmWindow(OK_MODE,"Can't set property!\n" + anErrorMessage,'Error!')
                return None

        # int
        if type(anOldValue) == int:
            try:
                aValue = int(aValue)
            except:
                # displays confirm window
                anErrorMessage = "Input int value."
                aDialog = ConfirmWindow(OK_MODE,"Can't set property!\n" + anErrorMessage,'Error!')
                return None

        # sets new value
        try:
            aStepperStub.setProperty( aPropertyName, aValue )
            aPropertyModel.set_value( iter, VALUE_INDEX, aValue )
        except:

            # displays error message to MessageWindow
            anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
            self.theSession.message( anErrorMessage )

            # displays confirm window
            anErrorMessage = "See MessageWindow for details."
            aDialog = ConfirmWindow(OK_MODE,aMessage,"Can't set property!\n" + anErrorMessage)
            return False

        # when to set property is succeeded,
        else:
            # refreshs self['property_list']
            self.update()
            return True

    # ==========================================================================
    def update( self ):
        """overwrites superclass's method
        updates the value of self['property_list']
        """

        # When this window does not created, does nothing
        if not self.exists():
            return None

        # gets stepperID
        iter_stepper = self['stepper_id_list'].get_selection().get_selected()[1]
        aStepperID = self['stepper_id_list'].get_model().get_value(iter_stepper,0)
        aStepperStub = self.theSession.createStepperStub( aStepperID )

        iter = self.theFirstPropertyIter[aStepperID]

        # updates all value of self['property_list']
        while( True ):
            iter = self['property_list'].get_model().iter_next(iter)
            if iter == None:
                break
            aProperty = self['property_list'].get_model().get_value(iter,0)
            if type ( self.thePropertyMap[ aProperty ] ) != type( () ):
                if aStepperStub.getPropertyAttributes( aProperty )[GETABLE]:
                    aValue = str( aStepperStub.getProperty( aProperty ) )
                else:
                    aValue = ''
                self['property_list'].get_model().set_value(iter,1,aValue)

        # updates text


