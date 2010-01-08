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
import copy
import os
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

class MultiPlotWindow( OsogoPluginWindow ):

    def __init__( self, dirname, data, pluginManager, rootWidget=None ):
        
        # calls superclass's constructor
        OsogoPluginWindow.__init__( self, dirname, data,
                                    pluginManager, rootWidget=rootWidget )

        self.data = None
        self.fullpnList = None
        self.defaultList = None
        self.statusbar = None

    # end of __init__

    def openWindow( self ):

        OsogoPluginWindow.openWindow( self )
        
        # add handers
        self.addHandlers( { 
                'on_save_button_clicked'  : self.__save,
                'on_view_button_clicked'  : self.__view,
                'on_close_button_clicked' : self.__close,
                'on_fullpn_box_changed'   : self.__selectFullPN,
                'on_add_button_clicked'   : self.__addState,
                'on_clear_button_clicked' : self.__clearStates,
                'on_targetpn_box_changed' : self.__changed, 
                'on_tmin_entry_changed'   : self.__changed, 
                'on_tmax_entry_changed'   : self.__changed, 
                'on_filter_entry_changed' : self.__filterChanged,
                'on_target_filter_entry_changed': self.__targetFilterChanged,
                } )

        self.statusbar = self[ 'statusbar' ]
        self.clearStatusbar()

        self.setIconList(
            os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )

        self.stateTree = self[ 'state_tree' ]
        self.__initializeStateTree()

        self.__initializeComboBoxes()
        if self.theSession.theModelWalker is not None:
            self.__updateFullPNBox()

            fullpnList = self.__updateTargetFullPNBox()
            selectedList = self.getRawFullPNList()
            if len( selectedList ) != 0:
                fullpn = createFullPNString( selectedList[ 0 ] )
                if fullpn in fullpnList:
                    i = fullpnList.index( fullpn )
                    self[ 'targetpn_box' ].set_active( i )


        self[ 'tmin_entry' ].set_text( '0.0' )
        self[ 'tmax_entry' ].set_text( '100.0' )

        self[ 'save_button' ].set_sensitive( 1 )
        self[ 'view_button' ].set_sensitive( 1 )
        self[ 'close_button' ].set_sensitive( 1 )

        # registers myself to PluginManager
        self.thePluginManager.appendInstance( self ) 

    def update( self ):

        if self.theSession.theModelWalker == None:
            return

    def getEntityProperty( self, fullpn ):

        return self.theSession.theSimulator.getEntityProperty( fullpn )

    def getEntityPropertyList( self, fullid ):

        return self.theSession.theSimulator.getEntityPropertyList( fullid )

    def getEntityPropertyAttributes( self, fullpn ):

        return self.theSession.theSimulator.getEntityPropertyAttributes( fullpn )

    def __initializeStateTree( self ):
        
        fullpnColumn = gtk.TreeViewColumn( 'FullPN',
                                           gtk.CellRendererText(),
                                           text=0 )
        fullpnColumn.set_reorderable( True )
        fullpnColumn.set_sort_column_id( 0 )
        self.stateTree.append_column( fullpnColumn )

        renderer = gtk.CellRendererText()
        renderer.connect( 'edited', self.__editState, 1 )
        renderer.set_property( 'editable', True )
        valueColumn = gtk.TreeViewColumn( 'Value', renderer, text=1 )
        valueColumn.set_reorderable( True )
        valueColumn.set_sort_column_id( 1 )
        self.stateTree.append_column( valueColumn )

        self.stateTree.set_search_column( 1 )

        model = gtk.ListStore( gobject.TYPE_STRING, 
                               gobject.TYPE_STRING )
        self.stateTree.set_model( model )

    def __editState( self, widget, path, new_text, col ):

        if self.theSession.theModelWalker == None:
            return

        model = self.stateTree.get_model()
        iter = model.get_iter( path )
        fullpn = model.get_value( iter, 0 )

        if new_text == '':
            model.remove( iter )
        else:
            try:
                value = float( new_text )
            except:
                message = '[%s] is not a float number.' % new_text
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return
            else:
                model.set_value( iter, col, new_text )
        
        self.__changed()

    def __clearStates( self, *arg ):

        model = self.stateTree.get_model()
        model.clear()
        self.__changed()

    def __addState( self, *arg ):

        model = self[ 'fullpn_box' ].get_model()
        i = self[ 'fullpn_box' ].get_active()
        if i < 0:
            return
        
        fullpn = model.get_value( model[ i ].iter, 0 )
        valueString = self[ 'value_entry' ].get_text()

        rangeList = valueString.split( ':' )
        if len( rangeList ) == 1:
            try:
                value = float( valueString )
            except:
                message = 'Invalid property value [%s].\n' % valueString
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return
            else:
                model = self.stateTree.get_model()
                model.append( ( fullpn, valueString ) )
        elif len( rangeList ) == 3:
            try:
                min_value = float( rangeList[ 0 ] )
                interval = float( rangeList[ 1 ] )
                max_value = float( rangeList[ 2 ] )
            except:
                message = 'Invalid property range [%s].\n' % valueString
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return
            else:
                if ( max_value - min_value ) * interval <= 0:
                    message = 'Invalid property range [%s].\n' % valueString
                    dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                    return

                model = self.stateTree.get_model()
                for value in numpy.arange( min_value, 
                                           max_value + interval, 
                                           interval ):
                    model.append( ( fullpn, str( value ) ) )
        else:
            message = 'Invalid property range [%s].\n' % valueString
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        self.__changed()

    def updateComboBox( self, combobox, textList ):

        model = combobox.get_model()
        model.clear()

        for text in textList:
            combobox.append_text( text )

        if len( textList ) > 0:
            combobox.set_active( 0 )

    def __initializeComboBoxes( self ):

        # initialize fullpn_box
        model = gtk.ListStore( gobject.TYPE_STRING )
        self[ 'fullpn_box' ].set_model( model )

        renderer = gtk.CellRendererText()
        self[ 'fullpn_box' ].pack_start( renderer )
        self[ 'fullpn_box' ].add_attribute( renderer, 'text', 0 )

        # initialize target_box
        model = gtk.ListStore( gobject.TYPE_STRING )
        self[ 'targetpn_box' ].set_model( model )

        renderer = gtk.CellRendererText()
        self[ 'targetpn_box' ].pack_start( renderer )
        self[ 'targetpn_box' ].add_attribute( renderer, 'text', 0 )

    def getDefaultFullPNList( self ):

        if self.defaultList is not None:
            return copy.copy( self.defaultList )

        fullpnList = []

        for fullid in self.theSession.getVariableList():
            fullpnList.append( '%s:Value' % fullid )
            fullpnList.append( '%s:MolarConc' % fullid )

        for fullid in self.theSession.getProcessList():
            fullpnList.append( '%s:Activity' % fullid )
            fullpnList.append( '%s:MolarActivity' % fullid )

        self.defaultList = copy.copy( fullpnList )
        return fullpnList

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
        return fullpnList

    def clearStatusbar( self ):

        self.showMessageOnStatusbar( '' )

    def showMessageOnStatusbar( self, message ):

        message = ', '.join( message.split( '\n' ) )
        self.statusbar.push( 1, message )

    def __save( self, *arg ):

        def callback( states, targetpn, tmin, tmax ):
            self.__openFileSelection( self.__saveView,
                                      states, targetpn, tmin, tmax )
        
        self.__analyze( callback )

    def __view( self, *arg ):

        try:
            import Gnuplot
        except:
            message = 'Module [Gnuplot] not found.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        def callback( states, targetpn, tmin, tmax ):
 
            g = Gnuplot.Gnuplot( persist=1 )
            g( 'set term wxt' )
            g( 'set xrange [%g:%g]' % ( 0.0, tmax - tmin ) )
            g.xlabel( 'Time' )
            g.ylabel( targetpn )

            gdataList = []
            for state, data in zip( states, self.data ):
                data = numpy.transpose( data )
                titleString = '%s=%g' % state
                gdata = Gnuplot.Data( data[ 0 ] - tmin, data[ 1 ],
                                      title=titleString,
                                      with_='lines' )
                gdataList.append( gdata )

            g.plot( *gdataList )

        self.__analyze( callback )

    def __analyze( self, callback ):

        ( states, targetpn, tmin, tmax ) = self.__getParameters()

        if states is None:
            return

        if len( states ) == 0:
            message = 'Add at least one state.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if tmin is None or tmax is None:
            message = 'Simulation times (%g, %g) are invalid.\n' \
                % ( tmin, tmax )
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if targetpn is None:
            message = 'No FullPN is selected.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        try:
            _ = createFullPN( targetpn )
        except: 
            message = 'FullPN [%s] is invalid.\n' % targetpn
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if tmin < 0 or tmax <= 0:
            message = 'Simulation times (%g, %g) must be positive.\n' \
                % ( tmin, tmax )
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if tmin >= tmax:
            message = 'tmax [%g] must be greater than tmin [%g].\n' \
                % ( tmax, tmin )
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if self.data is None:
            if len( states ) > 10:
                message = 'Would you like to test [%d] states?\nUse SessionManager script for a heavy task.' % len( states )
                dialog = ConfirmWindow( OKCANCEL_MODE, message, 'Confirm' )

                # when canceled, does nothing 
                if dialog.return_result() != OK_PRESSED:
                    return

            task = self.analysisGenerator( callback, states, targetpn, 
                                           tmin, tmax )
            gobject.idle_add( task.next )
        else:
            callback( states, targetpn, tmin, tmax )

    def analysisGenerator( self, callback, states, targetpn, tmin, tmax ):

        self[ 'save_button' ].set_sensitive( 0 )
        self[ 'view_button' ].set_sensitive( 0 )
        self[ 'close_button' ].set_sensitive( 0 )

        model = self.theSession.asModel()
        emlSupport = EmlSupport( None, model.asString().encode() )

        result = []

        for i, state in enumerate( states ):
            self.setFlaction( int( float( i ) / len( states ) * 100.0 ) )

            fullpn, value = state
            session = emlSupport.createSession()
            if tmin > 0.0: # tmin != 0.0
                session.run( tmin )

            session.theSimulator.setEntityProperty( fullpn, value )
            loggerStub = session.createLoggerStub( targetpn )
            loggerStub.create()

            session.run( tmax - tmin )
            result.append( loggerStub.getData() )

            yield True

        self.data = result
        self.setFlaction( 100 )
        yield True

        callback( states, targetpn, tmin, tmax )

        self[ 'save_button' ].set_sensitive( 1 )
        self[ 'view_button' ].set_sensitive( 1 )
        self[ 'close_button' ].set_sensitive( 1 )

        yield False

    def setFlaction( self, per ):

        if self[ 'progressbar' ] is not None:
            self[ 'progressbar' ].set_fraction( per * 0.01 )
            self[ 'progressbar' ].set_text( '%d%%' % per )

    def __changed( self, *arg ):

        self.data = None
        self.setFlaction( 0 )
        self.update()

    def __close( self, *arg ):

        self.close()

    def __getParameters( self ):

        states = []
        model = self.stateTree.get_model()
        for row in model:
            iter = row.iter
            fullpn = model.get_value( iter, 0 )

            try:
                createFullPN( fullpn )
            except: 
                message = 'FullPN [%s] is invalid.\n' % fullpn
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                states = None
                break

            try:
                value = float( model.get_value( iter, 1 ) )
            except:
                message = 'Invalid property value [%s] for [%s].\n' \
                    % ( value, fullpn )
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                states = None
                break

            states.append( ( fullpn, value ) ) 

        model = self[ 'targetpn_box' ].get_model()
        i = self[ 'targetpn_box' ].get_active()
        if i < 0:
            targetpn = None
        else:
            targetpn = model.get_value( model[ i ].iter, 0 )

        try:
            tmin = float( self[ 'tmin_entry' ].get_text() )
            tmax = float( self[ 'tmax_entry' ].get_text() )
        except:
            tmin = None
            tmax = None

        return states, targetpn, tmin, tmax

    def __selectFullPN( self, *arg ):

        model = self[ 'fullpn_box' ].get_model()
        i = self[ 'fullpn_box' ].get_active()
        if i < 0:
            return
        
        fullpn = model.get_value( model[ i ].iter, 0 )
        value = self.getEntityProperty( fullpn )

        self[ 'value_entry' ].set_text( str( value ) )

    def __openFileSelection( self, callback, *arg ):

        dialog = gtk.FileChooserDialog( "Open..", None,
                                        gtk.FILE_CHOOSER_ACTION_SAVE,
                                        ( gtk.STOCK_CANCEL, 
                                          gtk.RESPONSE_CANCEL,
                                          gtk.STOCK_OPEN, gtk.RESPONSE_OK ) )
        dialog.set_default_response( gtk.RESPONSE_OK )
   
        filter = gtk.FileFilter()
        extensions = [ 'csv', 'dat', 'eps', 'ps', 'png', 
                       'jpg', 'jpeg', 'svg', 'gif' ]
        filter.set_name( ', '.join( extensions ) )

        for ext in extensions:
            filter.add_mime_type( ext )
            filter.add_pattern( '*.%s' % ext )

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

            dialog.destroy()
            callback( filename, *arg )

        else: # response == gtk.RESPONSE_CANCEL
            dialog.destroy()

    def __saveView( self, filename, states, targetpn, tmin, tmax ):

        _, ext = os.path.splitext( filename )
        ext = ext.lower()

        if ext == '.csv':

            data = integrateData( self.data )
            csvWriter = csv.writer( file( filename, 'w' ), 
                                    lineterminator="\n" )

            labels = [ 't' ]
            labels.extend( [ '%s=%g' % state for state in states ] )
            csvWriter.writerow( labels )

            for stateArray in data:
                state = [ '%.16e' % x for x in stateArray ]
                csvWriter.writerow( state )
        
        elif ext == '.dat':

            data = integrateData( self.data )
            outputFile = open( filename, 'w' )

            labels = [ 't' ]
            labels.extend( [ '%s=%g' % state for state in states ] )
            outputFile.write( '# %s\n' % ( '\t'.join( labels ) ) )

            for stateArray in data:
                state = [ '%.16e' % x for x in stateArray ]
                outputFile.write( '%s\n' % ( '\t'.join( state ) ) )
        
        elif ext in [ '.eps', '.ps', '.png', '.jpg', '.jpeg', '.svg', '.gif' ]:
        
            try:
                import Gnuplot
            except:
                message = 'Module [Gnuplot] not found.\n'
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return

            g = Gnuplot.Gnuplot( persist=1 )

            if ext == '.eps':
                g( 'set term post enhanced color' )
            elif ext == '.ps':
                g( 'set term post' )
            elif ext == '.png':
                g( 'set term png' )
            elif ext == '.jpg' or ext == '.jpeg':
                g( 'set term jpeg' )
            elif ext == '.svg':
                g( 'set term svg' )
            elif ext == '.gif':
                g( 'set term gif' )

            g( "set output '%s'" % filename )

            g( 'set xrange [%g:%g]' % ( 0.0, tmax - tmin ) )
            g.xlabel( 'Time' )
            g.ylabel( targetpn )

            gdataList = []
            for state, data in zip( states, self.data ):
                data = numpy.transpose( data )
                titleString = '%s=%g' % state
                gdata = Gnuplot.Data( data[ 0 ] - tmin, data[ 1 ],
                                      title=titleString,
                                      with_='lines' )
                gdataList.append( gdata )

            g.plot( *gdataList )

        else:
            message = 'The extension [%s] is not supported (*.csv, *.dat, *.eps, *.ps, *.png, *.jpg, *.jpeg, *.svg, *.gif).\n' % ext
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        message = 'Multiple plots were saved as \n[%s].\n' % filename
        dialog = ConfirmWindow( OK_MODE, message, 'Succeeded!' )


    def filterFullPNList( self, fullpnList, entryName ):

        substringList = self[ entryName ].get_text().split()

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

    def __targetFilterChanged( self, *arg ):

        self.__updateTargetFullPNBox()

    def __filterChanged( self, *arg ):

        self.__updateFullPNBox()

    def __updateTargetFullPNBox( self ):

        if self.theSession.theModelWalker is None:
            return []

        fullpnList = self.filterFullPNList( self.getDefaultFullPNList(),
                                            'target_filter_entry' )

        filteredNum, fullpnNum \
            = len( fullpnList ), len( self.getDefaultFullPNList() )
        if fullpnNum < 1000:
            labelString = ' Filter (%d/%d):' % ( filteredNum, fullpnNum )
        elif fullpnNum < 1e+7:
            labelString = ' Filter (%d):' % filteredNum
        else:
            labelString = ' Filter:'

        self[ 'target_filter_label' ].set_label( labelString )

        self.updateComboBox( self[ 'targetpn_box' ], fullpnList )
        return fullpnList

    def __updateFullPNBox( self ):

        if self.theSession.theModelWalker is None:
            return []

        fullpnList = self.filterFullPNList( self.getFullPNList(),
                                            'filter_entry' )

        filteredNum, fullpnNum = len( fullpnList ), len( self.getFullPNList() )
        if fullpnNum < 1000:
            labelString = ' Filter (%d/%d):' % ( filteredNum, fullpnNum )
        elif fullpnNum < 1e+7:
            labelString = ' Filter (%d):' % filteredNum
        else:
            labelString = ' Filter:'

        self[ 'filter_label' ].set_label( labelString )

        self.updateComboBox( self[ 'fullpn_box' ], fullpnList )
