#!/usr/bin/env python

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-CELL Session Monitor package
#
#                Copyright (C) 1996-2002 Keio University
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
#'Design: Kenta Hashimoto <kem@e-cell.org>',
#'Design and application Framework: Kouichi Takahashi <shafi@e-cell.org>',
#'Programming: Yuki Fujita',
#             'Yoshiya Matsubara',
# 'Yuusuke Saito'
#
# modified by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-CELL Project, Lab. for Bioinformatics, Keio University.
#


import sys
import traceback
import os
import imp
import glob
from ecell.ECS import *
from config import *
#from Plugin import *
from ecell.Plugin import *

# ---------------------------------------------------------------
# OsogoPluginManager -> PluginManager
#   - sets the information about plugin modules to show interface
#     Interface Window
#   - updates the information on LoggerWindow
#   - sets the message to MessageWindow
#   - catchs the message from Session
# ---------------------------------------------------------------
class OsogoPluginManager(PluginManager):

	# ---------------------------------------------------------------
	# Constructor
	#   - alls the constructor of superclass 
	#   - sets the reference to some window
	#
	# aSession          : session
	# aLoggerWindow     : LoggerWindow
	# anInterfaceWindow : InterfaceWindow
	# aMessageWindow    : MessageWindow
	# return -> None
	# ---------------------------------------------------------------
	def __init__( self, aSession, aLoggerWindow, anInterfaceWindow, 
	              aMessageWindow  ):

		try:
			#self.thePluginMap = {}
			#self.theInstanceList = []

			PluginManager.__init__(self)

			self.theSession = aSession
			self.theLoggerWindow = aLoggerWindow
			self.theInterfaceWindow = anInterfaceWindow
			self.theMessageWindow = aMessageWindow
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			print aMessage

	# end of __init__
        

	# ---------------------------------------------------------------
	# createInstace (overrides PluginManager)
	#   - creates one plugin module
	#   - adds record to InterfaceWindow
	#   - call initialize method of the simulator of the session
	#
	# aClassName        : class name of plugin module
	# aData             : data 
	# aRoot             : None or 'menu' or 'top_vbox'
	# aParent           : parent window
	# return -> one instance
	# ---------------------------------------------------------------
	def createInstance( self, classname, data, root=None, parent=None ):
	

		try:
			try:
				# gets one plugin from plugin map
				aPlugin = self.thePluginMap[ classname ]
	    
				# If there is no plugin whose class is aClassName,
				# then call loadModule method of this class.
			except KeyError:
				self.loadModule( classname )

			# if the plugin module is not top module,
			# then add record to InterfaceWindow
			if root !='top_vbox':
				self.theInterfaceWindow.addNewRecord( classname, data )

			# creates instance
			#anInstance = aPlugin.createInstance( self, data, self, root, parent )

			# Nothing is selected.
			if len(data) == 0:
				self.printMessage("Nothing is selected.")

			else:

				anInstance = aPlugin.createInstance( data, self, root, parent )

				if root !='top_vbox':
					anInstance.editTitle( self.theInterfaceWindow.theTitle )

				# initialize session
				self.theSession.theSimulator.initialize()
				return anInstance

		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)


	# end of createInstance

	# ---------------------------------------------------------------
	# Loads a module (overrides PluginManager)
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# aClassName     : class name
	# return -> None 
	# ---------------------------------------------------------------
	def loadModule( self, aClassname ):
	
		try:
			PluginManager.loadModule(self,aClassname)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of loadModule

	# ---------------------------------------------------------------
	# loadAll (overrides PluginManager)
	#   - loads all plugin modules
	#   - adds record to InterfaceWindow
	#   - call initialize method of the simulator of the session
	#
	# return -> None
	# ---------------------------------------------------------------
	def loadAll( self ):

		try:
			for aPath in PLUGIN_PATH:
				aFileList = glob.glob( os.path.join( aPath, '*.glade' ) )
				for aFile in aFileList:
					aModulePath = os.path.splitext( aFile )[0]
					if( os.path.isfile( aModulePath + '.py' ) ):
						aModuleName = os.path.basename( aModulePath )
						self.loadModule( aModuleName )
						self.theInterfaceWindow.thePluginWindowsNoDict[ aModuleName[ : -6 ] ] = 0
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of loadAll



	# ---------------------------------------------------------------
	# updateAllPluginWindow (overrides PluginManager)
	#   - updates all plugin window
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# return -> None
	# ---------------------------------------------------------------
	def updateAllPluginWindow( self ):

		try:
			PluginManager.updateAllPluginWindow(self)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of updateAllPluginWindow


	# ---------------------------------------------------------------
	# appendInstance  (overrides PluginManager)
	#   - appends an instance to instance list
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# anInstance     : an instance
	# return -> None
	# ---------------------------------------------------------------
	def appendInstance( self, anInstance ):

		try:
			PluginManager.appendInstance(self, anInstance)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of appendInstance


	# ---------------------------------------------------------------
	# removeInstance (overrides PluginManager)
	#   - removes an instance from instance list
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# anInstance     : an instance
	# return -> None
	# This method is throwable exception. (ValueError)
	# ---------------------------------------------------------------
	def removeInstance( self, anInstance ):

		try:
			PluginManager.removeInstance(self, anInstance)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of removeInstance


	# ---------------------------------------------------------------
	# showPlugin (overrides PluginManager)
	#   - shows plugin window
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# aIndex       : an index of module
	# *Objects     : dammy elements of argument
	# return -> None
	# This method is throwable exception. (IndexError)
	# ---------------------------------------------------------------
	def showPlugin( self, num, obj ):

		try:
			PluginManager.showPlugin(self, anIndex, *Objects)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			aMessage += '%s' %traceback.print_exc()
			self.printMessage(aMessage)

	# end of showPlugin


	# ---------------------------------------------------------------
	# editModuleTitle (overrides PluginManager)
	#   - edits module title
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# anIndex    : index of instance
	# aTitle     : title of instance
	# return -> None
	# This method is throwable exception. (IndexError)
	# ---------------------------------------------------------------
	def editModuleTitile( self, anIndex, aTitle ):


		try:
			#self.theInstanceList[ anIndex + 1 ].editTitle( aTitle )
			WindowManager.editModuleTitle( self, anIndex, aTitle)
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of getModule


	# ---------------------------------------------------------------
	# deleteModule (overrides PluginManager)
	#   - deletes a module
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# anIndex     : index of instance
	# *Object     : dammy elements of argument
	# return -> None
	# This method is throwable exception. (IndexError)
	# ---------------------------------------------------------------
	def deleteModule( self, anIndex, *Objects ):

		try:
			anInstance = self.theInstanceList[ anIndex + 1 ]
			anInstance.getWidget( anInstance.__class__.__name__ ).destroy()
		except:
			aMessage  = '----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of deleteModule


	# ---------------------------------------------------------------
	# updateLoggerWindow 
	#   - call update method of LoggerWindow
	#
	# return -> None
	# ---------------------------------------------------------------
	def updateLoggerWindow( self ):

		try:
			self.theLoggerWindow.update()
		except:
			aMessage  = '\n----------Error------------\n'
			aMessage += 'ErroType[%s]\n'  %sys.exc_type
			aMessage += 'ErroValue[%s]\n' %sys.exc_value
			traceback.print_exc()
			self.printMessage(aMessage)

	# end of updateLoggerWindow

	# ---------------------------------------------------------------
	# printMessage
	#   - sets message to MessageWindow
	#
	# aMessage( string or list or tuple) : message will be shown on
	#                                      MessageWindow
	#
	# return -> None
	# ---------------------------------------------------------------
	def printMessage( self, aMessage ):

		self.theMessageWindow.printMessage(aMessage)

	# end of printMessage



if __name__ == "__main__":
    pass





