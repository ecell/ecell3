#!/usr/bin/env python
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2010 Keio University
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

"""
opengui.py  - to start up E-Cell3 GUI  
"""

__program__ = 'opengui.py'
__author__ = 'Koichi Takahashi <shafi@e-cell.org>, zhaiteng and sugi'
__copyright__ = 'Copyright (C) 2002-2008 Keio University'
__license__ = 'GPL'

import os
import sys
import getopt
import traceback

try:
	import pygtk
	pygtk.require('2.0')
	import gtk
	import gobject 
	from ecell.ui.osogo.GtkSessionMonitor import GtkSessionMonitor
except KeyboardInterrupt:
	sys.exit(1)

def loadScript( aTupple ):
	aSession = aTupple[0]
	anEssFile = aTupple[1]

	#loads script after main has been called
	try:
		# load ane execute script file
		aSession.loadScript( anEssFile )
	except:
		aSession.message(' can\'t load [%s]' %anEssFile)
		anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
		aSession.message("-----------")
		aSession.message(anErrorMessage)
		aSession.message("-----------")
	#else:
		# initialize & update windows
		#aSession.openWindow('MainWindow
		#aSession.updateWindow()






# -----------------------------------------------------------------
# main
#   - checks arguments
#   - creates GtkSessioMonitor instance 
#   - creates MainWindow instance
#   - executs options
#   - calls GUI_interact() 
# -----------------------------------------------------------------
def main():

	# -------------------------------------
	# initialize file names
	# -------------------------------------
	anEmlFile = None
	anEssFile = None
	anIniFile = None

	# -------------------------------------
	# gets options
	# -------------------------------------
	try:
		opts , args = getopt.getopt( sys.argv[1:] , 'he:f:i:',
					     ["help", "exec=", 
					      "file=", "ini="])
	except:
		usage()
		sys.exit(1)

	# -------------------------------------
	# checks argument
	# -------------------------------------
	for anOption, anArg in opts:

		# prints help message
		if anOption in ( "-h", '--help' ):
			usage()
			sys.exit(0)

		# executes script file (.ess)
		if anOption in ( "-e", '--exec'):
			anEssFile = anArg

		# load model file (.eml)
		if anOption in ( "-f", '--file' ):
			anEmlFile = anArg
			
		# load model file (.eml)
		if anOption in ( "-i", '--ini' ):
			anIniFile = anArg
			
	# -------------------------------------
	# prohibits to use -e and -f options 
	# -------------------------------------
	if anEmlFile != None and anEssFile != None:
		usage()
		sys.exit(0)

	# -------------------------------------
	# creates an instance of GtkSession and 
	# creates MainWindow instance
	# -------------------------------------
	aSession = GtkSessionMonitor()


	# -------------------------------------
	# executes options
	# -------------------------------------
	# load model file (.eml)
	if anEmlFile != None:

		# check EML File
		if os.path.isfile( anEmlFile ):
			pass
		else:
			aMessage = " Error ! [%s] No such file. \n" %anEmlFile
			aSession.message( aMessage )
			sys.exit(1)

		# print message 
		aSession.message("%s is loaded.\n" %anEmlFile )

		# load model
		try:
			aSession.loadModel( anEmlFile )
		except:
		
			aSession.message(' can\'t load [%s]' %anEmlFile)
			anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
			aSession.message("-----------")
			aSession.message(anErrorMessage)
			aSession.message("-----------")
		else:

			# initialize & update windows
			aMainWindow = aSession.openWindow('MainWindow')
			aSession.updateWindows()

			if anIniFile is not None \
				    and os.path.isfile( anIniFile ):
				# load model
				aSession.message( "%s is loaded.\n" 
						  % anIniFile )

				try:
					aMainWindow.setAppearance( anIniFile )
				except:
					aSession.message( ' can\'t load [%s]' 
							  % anIniFile )
					anErrorMessage = '\n'.join( traceback.format_exception( sys.exc_type,sys.exc_value,sys.exc_traceback ) )
					aSession.message( "-----------" )
					aSession.message( anErrorMessage )
					aSession.message( "-----------" )
				else:
					aSession.updateWindows()

	# executes script file (.ess)
	elif anEssFile != None:

		# check ESS File
		if os.path.isfile( anEssFile ):
			pass
		else:
			aMessage = " Error ! [%s] No such file. \n" %anEssFile
			aSession.message( aMessage )
			sys.exit(1)

		# print message on MainWindow
		aSession.message("%s is being loaded and executed.\n" %anEssFile )
		gobject.timeout_add( 1, loadScript, [aSession, anEssFile] )

	else:
		aMainWindow = aSession.openWindow('MainWindow')

	# -------------------------------------
	# calls gtk.main()
	# -------------------------------------

	aSession.GUI_interact()


# -----------------------------------------------------------------
# usage
#  - print usage 
# return -> None
# -----------------------------------------------------------------
def usage():
	aProgramName = 'ecell3-session-monitor'
	print '''
%(appname)s -- E-Cell3 Session Monitor

Usage:
    %(appname)s [-h] ([-e ESSFILE]|[-f EMLFILE])

Options:
    -e or --exec=[ESSFILE]  : Load a script (.ess) file on startup
    -f or --file=[EMLFILE]  : Load a model (.eml) file on startup
    -h or --help            : Print this message

    Either -e or -f option can be specified at once.

Configurations:
    If ECELL3_DM_PATH environment variable is set to a colon (%(pathsep)s)
    separated directory path, it tries to find dynamic modules within the
    locations referred to by it.

    Example: 
      ECELL3_DM_PATH=/home/user/dm%(pathsep)s/home/user/dm_other %(appname)s

''' % { 'appname': aProgramName, 'pathsep': os.pathsep }

