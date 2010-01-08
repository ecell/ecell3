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
import re
import zipfile
import cStringIO
import gtk
import gobject

from ecell.ecssupport import *
import ecell.ui.osogo.config as config
from ecell.ui.osogo.OsogoPluginWindow import *
import ecell.ecs_constants as ecs_constants

from ecell.analysis.emlsupport import EmlSupport
from ecell.analysis.Elasticity import getSensitivityArray
from ecell.analysis.ecdsupport import flexions
import numpy


DEFAULT_IGNORED_DICT \
    = { ecs_constants.VARIABLE: [ 'DiffusionCoeff', 
                                  'MolarConc', 'NumberConc' ], 
        ecs_constants.PROCESS: [ 'Priority', 'Activity' ],
        ecs_constants.SYSTEM: [] }

(CHECK_COLUMN, FULLPN_COLUMN) = range( 2 )

class BifurcationWindow( OsogoPluginWindow ):

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
                'on_view_button_clicked' : self.__view,
                'on_close_button_clicked' : self.__close,
                'on_property_box_changed' : self.__selectProperty,
                'on_property_tree_cursor_changed': self.__changed,
                'on_start_entry_changed': self.__changed,
                'on_end_entry_changed': self.__changed,
                'on_resolution_entry_changed': self.__changed,
                'on_freerun_entry_changed': self.__changed,
                'on_duration_entry_changed': self.__changed,
                'on_plot_all_check_toggled': self.__changed,
                'on_is_log_scale_toggled': self.__changed,
                'on_filter_entry_changed': self.__filterChanged,
                'on_plot_filter_entry_changed': self.__plotFilterChanged,
                } )

        self.propertyTree = self[ 'property_tree' ]
        self.__initializePropertyTree()
        self.__initializePropertyBox()
        if self.theSession.theModelWalker != None:
            self.removeRawFullPNList( 
                [ fullpn for fullpn in self.getRawFullPNList() 
                  if not fullpn in self.getDefaultFullPNList() ] )
            self.__updatePropertyTree()
            self.__updatePropertyBox()

        self[ 'freerun_entry' ].set_text( '1000.0' )
        self[ 'duration_entry' ].set_text( '100.0' )

        self.statusbar = self[ 'bif_statusbar' ]
        self.clearStatusbar()

        self.setIconList(
            os.path.join( config.GLADEFILE_PATH, "ecell.png" ),
            os.path.join( config.GLADEFILE_PATH, "ecell32.png" ) )

        self[ 'save_button' ].set_sensitive( 1 )
        self[ 'view_button' ].set_sensitive( 1 )
        self[ 'close_button' ].set_sensitive( 1 )

        # registers myself to PluginManager
        self.thePluginManager.appendInstance( self ) 

    def __initializePropertyTree( self ):
        
        renderer = gtk.CellRendererToggle()
        renderer.set_property( "activatable", True )
        renderer.connect( "toggled", self.toggleCheckbox )
        checkboxColumn = gtk.TreeViewColumn( '', renderer, 
                                             active=CHECK_COLUMN )
        checkboxColumn.set_reorderable( True )
        checkboxColumn.set_sort_column_id( 0 )
        self.propertyTree.append_column( checkboxColumn )

        fullpnColumn = gtk.TreeViewColumn( 'FullPN',
                                           gtk.CellRendererText(),
                                           text=FULLPN_COLUMN )
        fullpnColumn.set_reorderable( True )
        fullpnColumn.set_sort_column_id( 1 )
        self.propertyTree.append_column( fullpnColumn )

        self.propertyTree.set_search_column( 1 )

        model = gtk.ListStore( gobject.TYPE_BOOLEAN,
                               gobject.TYPE_STRING )
        self.propertyTree.set_model( model )

    def update( self ):

        if self.theSession.theModelWalker == None:
            return

    def getEntityProperty( self, fullpn ):

        return self.theSession.theSimulator.getEntityProperty( fullpn )

    def getEntityPropertyList( self, fullid ):

        return self.theSession.theSimulator.getEntityPropertyList( fullid )

    def getEntityPropertyAttributes( self, fullpn ):

        return self.theSession.theSimulator.getEntityPropertyAttributes( fullpn )

    def getDefaultFullPNList( self ):

        if self.defaultList is not None:
            return copy.copy( self.defaultList )

        fullpnList = []
        for fullid in self.theSession.getVariableList():
            fullpnList.append( '%s:Value' % fullid )
            fullpnList.append( '%s:MolarConc' % fullid )
            fullpnList.append( '%s:NumberConc' % fullid )

        for fullid in self.theSession.getProcessList():
            fullpnList.append( '%s:Activity' % fullid )
            fullpnList.append( '%s:MolarActivity' % fullid )

        self.defaultList = copy.copy( self.fullpnList )
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

        self.fullpnList = copy.copy( self.fullpnList )
        return fullpnList

    def __initializePropertyBox( self ):

        model = gtk.ListStore( gobject.TYPE_STRING )
        self[ 'property_box' ].set_model( model )

        renderer = gtk.CellRendererText()
        self[ 'property_box' ].pack_start( renderer )
        self[ 'property_box' ].add_attribute( renderer, 'text', 0 )

    def clearStatusbar( self ):

        self.showMessageOnStatusbar( '' )

    def showMessageOnStatusbar( self, message ):

        message = ', '.join( message.split( '\n' ) )
        self.statusbar.push( 1, message )

    def setFlaction( self, per ):

        if self[ 'bif_progressbar' ] is not None:
            self[ 'bif_progressbar' ].set_fraction( per * 0.01 )
            self[ 'bif_progressbar' ].set_text( '%d%%' % per )

    def __save( self, *arg ):

        def callback( fullpnList, targetpn, startValue, endValue, 
                      resolution, freerun, duration, filtered, logscale ):
            self.__openFileSelection( self.__saveBifurcationDiagram, 
                                      fullpnList, targetpn,
                                      startValue, endValue, logscale )
        
        self.__analyze( callback )

    def __view( self, *arg ):

        try:
            import Gnuplot
        except:
            message = 'Module [Gnuplot] not found.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        def callback( fullpnList, targetpn, startValue, endValue, 
                      resolution, freerun, duration, filtered, logscale ):
 
            g = Gnuplot.Gnuplot( persist=1 )
            g( 'set term wxt' )
            g( 'set xrange [%g:%g]' % ( startValue, endValue ) )
            g.xlabel( targetpn )
            if logscale:
                g( 'set log x' )
#             else:
#                 g( 'unset log x' )

            pnList = []
            for fullpn in fullpnList:
                propertyName = createFullPN( fullpn )[ -1 ] 
                if not propertyName in pnList:
                    pnList.append( propertyName )

            if len( pnList ) > 1:
                ylabel = '%s and %s' % ( ', '.join( pnList[ 1 : ] ), 
                                         pnList[ -1 ] )
                g.ylabel( ylabel )
            else:
                g.ylabel( pnList[ 0 ] )

            gdataList = []
            for fullpn, pair in self.data.items():
                data = numpy.transpose( pair[ 1 ] )
                gdata = Gnuplot.Data( data[ 0 ], data[ 1 ],
                                      title=fullpn,
                                      with_=pair[ 0 ] )
                gdataList.append( gdata )

            g.plot( *gdataList )

        self.__analyze( callback )

    def __analyze( self, callback ):

        ( fullpnList, targetpn, 
          startValue, endValue, resolution, freerun, duration, 
          filtered, logscale ) \
          = self.__getParameters()

        if len( fullpnList ) == 0:
            message = 'No properties are selected.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if targetpn is None:
            message = 'No parameter is selected.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return            

        if startValue is None \
                or endValue is None \
                or resolution is None \
                or freerun is None \
                or duration is None:
            message = 'Simulation paraters are invalid.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if endValue == startValue or resolution <= 0:
            message = 'Parameter values are invalid.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if logscale and ( startValue == 0 or endValue == 0 ):
            message = 'Parameters must be positive for log scale.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return            

        if freerun < 0 or duration < 0:
            message = 'Running time must be positive.\n'
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        if self.data is None:
            if resolution > 500:
                message = 'Would you like to run the analysis with Resolution [%d] over 500?\nUse SessionManager script for a heavy task.' % resolution
                dialog = ConfirmWindow( OKCANCEL_MODE, message,
                                        'Confirm Resolution' )

                # when canceled, does nothing 
                if dialog.return_result() != OK_PRESSED:
                    return

            task = self.analysisGenerator( callback, fullpnList, targetpn, 
                                           startValue, endValue, resolution, 
                                           freerun, duration, 
                                           filtered, logscale )
            gobject.idle_add( task.next )
        else:
            callback( fullpnList, targetpn, startValue, endValue, 
                      resolution, freerun, duration, filtered, logscale )

    def __getParameters( self ):

        fullpnList = []

#         model = self.propertyTree.get_model()
#         for row in model:
#             iter = row.iter
#             selected = model.get_value( iter, CHECK_COLUMN )
#             if selected:
#                 fullpn = model.get_value( iter, FULLPN_COLUMN )
#                 fullpnList.append( fullpn )

        fullpnList = [ createFullPNString( fullpn )
                       for fullpn in self.getRawFullPNList() ]

        i = self[ 'property_box' ].get_active()
        if i < 0:
            targetpn = None
        else:
            model = self[ 'property_box' ].get_model()
            targetpn = model.get_value( model[ i ].iter, 0 )

        try:
            startValue = float( self[ 'start_entry' ].get_text() )
            endValue = float( self[ 'end_entry' ].get_text() )
            resolution = float( self[ 'resolution_entry' ].get_text() )
            freerun = float( self[ 'freerun_entry' ].get_text() )
            duration = float( self[ 'duration_entry' ].get_text() )
        except:
            startValue = None
            endValue = None
            resolution = None
            freerun = None
            duration = None

        filtered = not self['plot_all_checkbox'].get_active()
        logscale = self['is_log_scale'].get_active()

        return fullpnList, targetpn, \
            startValue, endValue, resolution, freerun, duration, \
            filtered, logscale

    def __close( self, *arg ):

        self.close()

    def __changed( self, *arg ):

        self.data = None
        self.setFlaction( 0 )
        self.update()

    def __selectProperty( self, *arg ):

        i = self[ 'property_box' ].get_active()
        if i < 0:
            return

        model = self[ 'property_box' ].get_model()
        iter = model[ i ].iter
        fullpn = model.get_value( iter, 0 )

        value = self.getEntityProperty( fullpn )

        self[ 'start_entry' ].set_text( '0.0' )
        self[ 'end_entry' ].set_text( str( value ) )
        self[ 'resolution_entry' ].set_text( '100' )

        self.__changed( *arg )

    def analysisGenerator( self, callback, fullpnList, targetpn, 
                           startValue, endValue, 
                           resolution, freerun, duration, filtered, logscale ):

        self[ 'save_button' ].set_sensitive( 0 )
        self[ 'view_button' ].set_sensitive( 0 )
        self[ 'close_button' ].set_sensitive( 0 )

        model = self.theSession.asModel()
        emlSupport = EmlSupport( None, model.asString().encode() )

        result = {}
        for fullpn in fullpnList:
            result[ fullpn ] = [ 'lines', [] ]

        if logscale:
            startLogValue = numpy.log10( startValue )
            endLogValue = numpy.log10( endValue )
            interval = ( endLogValue - startLogValue ) / resolution
            valueRange = numpy.arange( startLogValue, endLogValue + interval, 
                                       interval )
            valueRange = 10 ** valueRange
        else:
            interval = ( endValue - startValue ) / resolution
            valueRange = numpy.arange( startValue, endValue + interval, 
                                       interval )            

        for cnt, value in enumerate( valueRange ):
            self.setFlaction( ( 100 * cnt ) / len( valueRange ) )

            session = emlSupport.createSession()
            session.theSimulator.setEntityProperty( targetpn, value )
            session.run( freerun )

            loggerList = []
            for fullpn in fullpnList:
                loggerList.append( session.createLoggerStub( fullpn ) )
                loggerList[ -1 ].create()

            if duration == 0:
                for fullpn in fullpnList:
                    x = session.theSimulator.getEntityProperty( fullpn )
                    data = numpy.array( [ value, x ] )
                    result[ fullpn ][ 1 ].append( data )
                continue

            session.run( duration )

            if filtered:
                for fullpn, logger in zip( fullpnList, loggerList ):
                    flexionList = flexions( logger.getData() )
                    flexionList = [ flexion[ 1 ] for flexion in flexionList ]
                    flexionList = numpy.unique( flexionList ) # numarray

                    if len( flexionList ) == 0:
                        flexionList \
                            = numpy.array( [ logger.getData()[ -1 ][ 1 ] ] )

                    if len( flexionList ) == 1:
                        data = numpy.array( [ value, flexionList[ 0 ] ] )
                        result[ fullpn ][ 1 ].append( data )
                    else:
                        result[ fullpn ][ 0 ] = 'points pt 6'
                        for x in flexionList:
                            data = numpy.array( [ value, x ] )
                            result[ fullpn ][ 1 ].append( data )
            else:
                for fullpn, logger in zip( fullpnList, loggerList ):
                    pointList = logger.getData()
                    result[ fullpn ][ 0 ] = 'dots'
                    for i in range( len( pointList ) ):
                        data = numpy.array( [ value, 
                                              pointList[ i ][ 1 ] ] )
                        result[ fullpn ][ 1 ].append( data )

            yield True

        self.setFlaction( 100 )

        for fullpn in fullpnList:
            result[ fullpn ][ 1 ] = numpy.array( result[ fullpn ][ 1 ] )

        self.data = result

        yield True

        callback( fullpnList, targetpn, startValue, endValue, 
                  resolution, freerun, duration, filtered, logscale )

        self[ 'save_button' ].set_sensitive( 1 )
        self[ 'view_button' ].set_sensitive( 1 )
        self[ 'close_button' ].set_sensitive( 1 )

        yield False

    def __openFileSelection( self, callback, *arg ):

        dialog = gtk.FileChooserDialog( "Open..", None,
                                        gtk.FILE_CHOOSER_ACTION_SAVE,
                                        ( gtk.STOCK_CANCEL, 
                                          gtk.RESPONSE_CANCEL,
                                          gtk.STOCK_OPEN, gtk.RESPONSE_OK ) )
        dialog.set_default_response( gtk.RESPONSE_OK )
   
        filter = gtk.FileFilter()
        extensions = [ 'zip', 'eps', 'ps', 'png', 
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

    def __saveBifurcationDiagram( self, filename, fullpnList, targetpn, 
                                  startValue, endValue, logscale ):

        _, ext = os.path.splitext( filename )
        ext = ext.lower()

        if ext == '.zip':

            archive = zipfile.ZipFile( filename, 'w', 
                                       zipfile.ZIP_DEFLATED )

            for fullpn, pair in self.data.items():
                output = cStringIO.StringIO()
                csvWriter = csv.writer( output, lineterminator="\n" )
                
                for state in pair[ 1 ]:
                    csvWriter.writerow( [ '%.16e' % state[ 0 ],
                                          '%.16e' % state[ 1 ] ] )

                outputfilename = '%s.csv' \
                    % ( fullpn.replace( ':', '_' ).replace( '/', '_' ) )
                info = zipfile.ZipInfo( outputfilename )
                info.external_attr = 0644 << 16L
                archive.writestr( info, output.getvalue() )
                output.close()

            archive.close()

        elif ext in [ '.eps', '.ps', '.png', '.jpg', '.jpeg', '.svg', '.gif' ]:
        
            try:
                import Gnuplot
            except:
                message = 'Module [Gnuplot] not found.\n'
                dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
                return

            g = Gnuplot.Gnuplot()

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

            g( 'set xrange [%g:%g]' % ( startValue, endValue ) )
            g.xlabel( targetpn )

            if logscale:
                g( 'set log x' )
#             else:
#                 g( 'unset log x' )

            pnList = []
            for fullpn in fullpnList:
                propertyName = createFullPN( fullpn )[ -1 ] 
                if not propertyName in pnList:
                    pnList.append( propertyName )

            if len( pnList ) > 1:
                ylabel = '%s and %s' % ( ', '.join( pnList[ 1 : ] ), 
                                         pnList[ -1 ] )
                g.ylabel( ylabel )
            else:
                g.ylabel( pnList[ 0 ] )

            gdataList = []
            for fullpn, pair in self.data.items():
                data = numpy.transpose( pair[ 1 ] )
                gdata = Gnuplot.Data( data[ 0 ], data[ 1 ],
                                      title=fullpn,
                                      with_=pair[ 0 ] )
                gdataList.append( gdata )

            g.plot( *gdataList )

        else:
            message = 'The extension [%s] is not supported (*.zip, *.eps, *.ps, *.png, *.jpg, *.jpeg, *.svg, *.gif).\n' % ext
            dialog = ConfirmWindow( OK_MODE, message, 'Error!' )
            return

        message = 'Bifurcation diagram was saved as \n[%s].\n' % filename
        dialog = ConfirmWindow( OK_MODE, message, 'Succeeded!' )

    def toggleCheckbox( self, cell, path ):

        model = self.propertyTree.get_model()
        iter = model.get_iter( path )
        selected = model.get_value( iter, CHECK_COLUMN )
        model.set_value( iter, CHECK_COLUMN, not selected )

        fullpn = model.get_value( iter, FULLPN_COLUMN )
        fullpn = createFullPN( fullpn )

        if fullpn in self.getRawFullPNList():
            if selected:
                self.removeRawFullPNList( [ fullpn ] )
        else:
            if not selected:
                self.appendRawFullPNList( [ fullpn ] )

    def filterFullPNList( self, fullpnList, entryName ):

        if self[ entryName ] is None:
            return fullpnList

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

    def __filterChanged( self, *arg ):

        self.__updatePropertyBox()

    def __plotFilterChanged( self, *arg ):

        self.__updatePropertyTree()

    def __updatePropertyBox( self ):

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

        self.updateComboBox( self[ 'property_box' ], fullpnList )        

    def __updatePropertyTree( self ):

        model = self.propertyTree.get_model()
        model.clear()

        fullpnList = self.filterFullPNList( self.getDefaultFullPNList(),
                                            'plot_filter_entry' )
        filteredNum, fullpnNum \
            = len( fullpnList ), len( self.getDefaultFullPNList() )
        if fullpnNum < 1000:
            labelString = ' Filter (%d/%d):' % ( filteredNum, fullpnNum )
        elif fullpnNum < 1e+7:
            labelString = ' Filter (%d):' % filteredNum
        else:
            labelString = ' Filter:'

        self[ 'plot_filter_label' ].set_label( labelString )

        for fullpn in fullpnList:
            model.append( ( False, fullpn ) )

        for row in model:
            fullpn = model.get_value( row.iter, FULLPN_COLUMN )
            fullpn = createFullPN( fullpn )
            if fullpn in self.getRawFullPNList():
                model.set_value( row.iter, CHECK_COLUMN, True )

    def updateComboBox( self, combobox, textList ):

        model = combobox.get_model()
        model.clear()

        for text in textList:
            combobox.append_text( text )

        if len( textList ) > 0:
            combobox.set_active( 0 )
