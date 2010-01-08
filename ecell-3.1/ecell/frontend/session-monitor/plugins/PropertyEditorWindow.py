#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2009 Keio University
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

import csv
import os
import copy
import os.path
import zipfile
import cStringIO
import gtk
import gobject
import re

from ecell.ecssupport import *
import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoPluginWindow import *
import ecell.ecs_constants as ecs_constants

from ecell.analysis.emlsupport import EmlSupport
from ecell.analysis.ecdsupport import integrateData
import numpy


DEFAULT_IGNORED_DICT \
    = { ecs_constants.VARIABLE: [ 'DiffusionCoeff', 
                                  'MolarConc', 'NumberConc' ], 
        ecs_constants.PROCESS: [ 'Priority', 'Activity' ],
        ecs_constants.SYSTEM: [] }

class PropertyEditorWindow( OsogoPluginWindow ):

    def __init__( self, dirname, data, pluginManager, rootWidget=None ):
        
        # calls superclass's constructor
        OsogoPluginWindow.__init__( self, dirname, data,
                                    pluginManager, rootWidget=rootWidget )

        self.inits = []
        for fullpn in data:
            if len( fullpn ) != 4:
                break
            else:
                self.inits.append( createFullPNString( fullpn ) )

        self.fullpnList = None
        self.bufferArray = None
        self.statusbar = None

    # end of __init__

    def openWindow( self ):

        OsogoPluginWindow.openWindow( self )
        
        # add handers
        self.addHandlers( { 
                'on_filter_entry_changed'   : self.__updatePropertyTree,
                'on_set_button_clicked'     : self.__setClicked,
                'on_save_button_clicked'    : self.__saveClicked,
                'on_revert_button_clicked'  : self.__revertClicked,
                'on_save_csv_button_clicked'  : self.__savePropertiesAsCSV,
                'on_load_csv_button_clicked'  : self.__loadPropertiesFromCSV,
                'on_multiply_button_clicked'  : self.__multiplyClicked,
                } )

        self.statusbar = self[ 'statusbar' ]
        self.clearStatusbar()

        self.setIconList(
            os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )
        self.propertyTree = self[ 'property_tree' ]
        self.__initializePropertyTree()
        if self.theSession.theModelWalker != None:
            self.__updatePropertyTree()

        self.updateButtons()

        # registers myself to PluginManager
        self.thePluginManager.appendInstance( self ) 

    def updateButtons( self ):

        if self.theSession.theModelWalker != None:
            self[ 'set_button' ].set_sensitive( 1 )
            self[ 'save_button' ].set_sensitive( 1 )
            self[ 'revert_button' ].set_sensitive( self.bufferArray 
                                                   is not None )
        else:
            self[ 'set_button' ].set_sensitive( 0 )
            self[ 'save_button' ].set_sensitive( 0 )
            self[ 'revert_button' ].set_sensitive( 0 )

    def update( self ):

        self.updateButtons()
        self.__updatePropertyTree()

    def __initializePropertyTree( self ):
        
        fullpnColumn = gtk.TreeViewColumn( 'FullPN',
                                           gtk.CellRendererText(),
                                           text=0 )
        fullpnColumn.set_reorderable( True )
        fullpnColumn.set_sort_column_id( 0 )
        self.propertyTree.append_column( fullpnColumn )

        renderer = gtk.CellRendererText()
        renderer.connect( 'edited', self.__editValue, 1 )
        renderer.set_property( 'editable', True )
        valueColumn = gtk.TreeViewColumn( 'Value', renderer, text=1 )
        valueColumn.set_reorderable( True )
        valueColumn.set_sort_column_id( 1 )
        self.propertyTree.append_column( valueColumn )

        renderer = gtk.CellRendererText()
#         renderer.connect( 'edited', self.__editValue, 1 )
#         renderer.set_property( 'editable', True )
        bufferColumn = gtk.TreeViewColumn( 'Buffer', renderer, text=2 )
        bufferColumn.set_reorderable( True )
        bufferColumn.set_sort_column_id( 2 )
        self.propertyTree.append_column( bufferColumn )

        self.propertyTree.set_search_column( 1 )

        model = gtk.ListStore( gobject.TYPE_STRING, 
                               gobject.TYPE_STRING,
                               gobject.TYPE_STRING )
        self.propertyTree.set_model( model )

    def __updatePropertyTree( self, *arg ):

        if self.theSession.theModelWalker is None:
            return

        model = self.propertyTree.get_model()
        model.clear()

        fullpnList = self.filterFullPNList( self.getFullPNList() )
        filteredNum, fullpnNum = len( fullpnList ), len( self.getFullPNList() )
        if fullpnNum < 1000:
            labelString = ' Filter (%d/%d):' % ( filteredNum, fullpnNum )
        elif fullpnNum < 1e+7:
            labelString = ' Filter (%d):' % filteredNum
        else:
            labelString = ' Filter:'

        self[ 'filter_label' ].set_label( labelString )

        if self.bufferArray is None:
            for fullpn in fullpnList:
                value = self.getEntityProperty( fullpn )
                model.append( ( fullpn, str( value ), '' ) )
        else:
            for i, fullpn in enumerate( fullpnList ):
                value = self.getEntityProperty( fullpn )
                model.append( ( fullpn, str( value ), self.bufferArray[ i ] ) )

    def __editValue( self, widget, path, new_text, col ):

        if self.theSession.theModelWalker == None:
            return

        model = self.propertyTree.get_model()
        iter = model.get_iter( path )
        fullpn = model.get_value( iter, 0 )

        if new_text != '':
            try:
                value = float( new_text )
            except:
                message = '[%s] is not a float number.' % new_text
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return
            else:
                self.setValue( createFullPN( fullpn ), value )
        
    def filterFullPNList( self, fullpnList ):

        substringList = self[ 'filter_entry' ].get_text().split()

        if len( substringList ) != 0:
            filterList = []
            for substring in substringList:
                try:
                    filter = re.compile( substring )
                    filterList.append( filter )
                except:
                    pass

            filter_func = lambda x: all( [ filter.search( x ) is not None 
                                           for filter in filterList ] )
            fullpnList = [ fullpn for fullpn in fullpnList 
                           if filter_func( fullpn ) ]

        return fullpnList

    def getEntityProperty( self, fullpn ):

        return self.theSession.theSimulator.getEntityProperty( fullpn )

    def getEntityPropertyList( self, fullid ):

        return self.theSession.theSimulator.getEntityPropertyList( fullid )

    def getEntityPropertyAttributes( self, fullpn ):

        return self.theSession.theSimulator.getEntityPropertyAttributes( fullpn )

    def __setClicked( self, *arg ):

        valueString = self[ 'value_entry' ].get_text()
        
        try:
            value = float( valueString )
        except:
            message = 'Property value [%s] must be a float number.\n' \
                % valueString
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        fullpnList = self.filterFullPNList( self.getFullPNList() )
        for fullpn in fullpnList:
            self.theSession.theSimulator.setEntityProperty( fullpn, value )
#             self.setValue( createFullPN( fullpn ), value )

        self.thePluginManager.updateAllPluginWindow()
        self.thePluginManager.updateFundamentalWindows()

    def __multiplyClicked( self, *arg ):

        valueString = self[ 'value_entry' ].get_text()
        
        try:
            value = float( valueString )
        except:
            message = 'Property value [%s] must be a float number.\n' \
                % valueString
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        fullpnList = self.filterFullPNList( self.getFullPNList() )
        for fullpn in fullpnList:
            x = self.getEntityProperty( fullpn )
            self.theSession.theSimulator.setEntityProperty( fullpn, x * value )

        self.thePluginManager.updateAllPluginWindow()
        self.thePluginManager.updateFundamentalWindows()

    def __saveClicked( self, *arg ):

        if self.bufferArray is not None:
            message = 'Would you like to replace the existing buffer? \n'
            dialog = ConfirmWindow( OKCANCEL_MODE, message, 
                                    'Confirm Buffer Overwrite' )
            if dialog.return_result() != OK_PRESSED:
                return

        fullpnList = self.getFullPNList()
        self.bufferArray = numpy.zeros( len( fullpnList ), float )
        for i, fullpn in enumerate( fullpnList ):
            self.bufferArray[ i ] = self.getEntityProperty( fullpn )

        self.update()

    def __revertClicked( self, *arg ):

        if self.bufferArray is None:
            return # never get here

        fullpnList = self.getFullPNList()
        for i, fullpn in enumerate( fullpnList ):
            value = self.bufferArray[ i ]
            self.theSession.theSimulator.setEntityProperty( fullpn, value )
#             self.setValue( createFullPN( fullpn ), self.bufferArray[ i ] )

        self.thePluginManager.updateAllPluginWindow()
        self.thePluginManager.updateFundamentalWindows()

    def __savePropertiesAsCSV( self, *arg ):

        dialog = gtk.FileChooserDialog( "Open..", None,
                                        gtk.FILE_CHOOSER_ACTION_SAVE,
                                        ( gtk.STOCK_CANCEL, 
                                          gtk.RESPONSE_CANCEL,
                                          gtk.STOCK_OPEN, gtk.RESPONSE_OK ) )
        dialog.set_default_response( gtk.RESPONSE_OK )
   
        filter = gtk.FileFilter()
        filter.set_name( "CSV" )
        filter.add_mime_type( "csv" )
        filter.add_pattern( "*.csv" )
        dialog.add_filter( filter )
   
        filter = gtk.FileFilter()
        filter.set_name( "All files" )
        filter.add_pattern( "*" )
        dialog.add_filter( filter )
   
        response = dialog.run()

        if response == gtk.RESPONSE_OK:
            filename = dialog.get_filename()

            # when the file already exists
            if os.path.isfile( filename ):
                message = 'Would you like to replace the existing file? \n[%s]' % filename
                confirmDialog = ConfirmWindow( OKCANCEL_MODE, message,
                                               'Confirm File Overwrite' )

                # when canceled, does nothing
                if confirmDialog.return_result() != OK_PRESSED:
                    dialog.destroy()
                    return

            self.theSession.savePropertiesAsCSV( filename )

        elif response == gtk.RESPONSE_CANCEL:
            dialog.destroy()
            return

        dialog.destroy()

        self.thePluginManager.updateAllPluginWindow()
        self.thePluginManager.updateFundamentalWindows()

    def __loadPropertiesFromCSV( self, *arg ):

        dialog = gtk.FileChooserDialog( "Open..", None,
                                        gtk.FILE_CHOOSER_ACTION_OPEN,
                                        ( gtk.STOCK_CANCEL, 
                                          gtk.RESPONSE_CANCEL,
                                          gtk.STOCK_OPEN, gtk.RESPONSE_OK ) )
        dialog.set_default_response( gtk.RESPONSE_OK )
   
        filter = gtk.FileFilter()
        filter.set_name( "CSV" )
        filter.add_mime_type( "csv" )
        filter.add_pattern( "*.csv" )
        dialog.add_filter( filter )
   
        filter = gtk.FileFilter()
        filter.set_name( "All files" )
        filter.add_pattern( "*" )
        dialog.add_filter( filter )
   
        response = dialog.run()

        if response == gtk.RESPONSE_OK:
            filename = dialog.get_filename()
            if not os.path.isfile( filename ):
                message = 'File [%s] not found.' % filename
                confirmDialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                dialog.destroy()
                return
            else:
                self.theSession.loadPropertiesFromCSV( filename )

        elif response == gtk.RESPONSE_CANCEL:
            dialog.destroy()
            return

        dialog.destroy()

        self.thePluginManager.updateAllPluginWindow()
        self.thePluginManager.updateFundamentalWindows()

    def getFullPNList( self, ignored=DEFAULT_IGNORED_DICT ):

        if self.fullpnList is not None:
            return copy.copy( self.fullpnList )

        fullidList = self.theSession.getVariableList()
        fullidList.extend( self.theSession.getProcessList() )

        fullpnList = []
        for fullid in fullidList:
            entityType = createFullID( fullid )[ 0 ]
            propertyList = self.getEntityPropertyList( fullid )

            for propertyName in propertyList:
                if propertyName in ignored[ entityType ]:
                    continue

                fullpn = '%s:%s' % ( fullid, propertyName )
                attributes = self.getEntityPropertyAttributes( fullpn )

                # 0: set, 1: get, 2: load, 3: save
                if attributes[ 0 ] != 0 and attributes[ 1 ] != 0:
                    value = self.getEntityProperty( fullpn )
                    if type( value ) == float:
                        fullpnList.append( fullpn )

        self.fullpnList = copy.copy( fullpnList )
        self.bufferArray = None
        return fullpnList

    def clearStatusbar( self ):

        self.showMessageOnStatusbar( '' )

    def showMessageOnStatusbar( self, message ):

        message = ', '.join( message.split( '\n' ) )
        self.statusbar.push( 1, message )
