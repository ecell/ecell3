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
#             'Yuusuke Saito'
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
from ecell.Plugin import *


class OsogoPluginManager(PluginManager):
	"""PluginManager specified for Osogo
	"""

	# ========================================================================
	def __init__( self, aMainWindow ):
		"""Constructor
		aMainWindow   ---  the instance of MainWindow (MainWindow)
		"""

		PluginManager.__init__(self)

		self.theSession = aMainWindow.theSession
		self.theMainWindow = aMainWindow

		self.thePluginTitleDict = {}     # key is instance , value is title
		self.thePluginWindowNumber = {}
		self.thePropertyWindowOnEntityListWindows = {}  # key is instance, value is None

	# end of __init__
        

	# ========================================================================
	def createInstance( self, aClassname, data, root=None, parent=None ):
		"""creates new Plugin Instance
		aClassName  --- a class name of PluginWindow (str)
		aData       --- a RawFullPN (RawFullPN)
		aRoot       --- a root widget (str) 
		aParent     --- ParentWindow (Window)          # NOT gtk.Window
		Returns a PluginWindow instance (PluginWindow)
		"""
	
		if self.thePluginMap.has_key( aClassname ):
			pass
		else:
			self.loadModule( aClassname )

		aPlugin = self.thePluginMap[ aClassname ]

		# -------------------------------------------------------
		# If plugin window does not exist on EntryList,
		# then creates new title and sets it to plugin window.
		# -------------------------------------------------------
		aTitle = ""
		#if root !='top_vbox':                # if(1)
		#if root !='EntityListWindow':                # if(1)


		#if root == None:                # if(1)
		#if parent == None:                # if(1)
		if root != None and root.__class__.__name__ == 'EntityListWindow':
			pass
		else:
			aTitle = aClassname[:-6]

			if self.thePluginWindowNumber.has_key( aClassname ):
				self.thePluginWindowNumber[ aClassname ] += 1
			else:
				self.thePluginWindowNumber[ aClassname ] = 1

			aTitle = "%s%d" %(aTitle,self.thePluginWindowNumber[ aClassname ])

			# if(1)


		# Nothing is selected.
		if len(data) == 0:
			self.printMessage("Nothing is selected.")

		else:

			try:
				anInstance = aPlugin.createInstance( data, self, root )
			except TypeError:
				return None

			anInstance.openWindow()

			#try:
			if TRUE:
				#if root !='top_vbox':              
				#if root != 'EntityListWindow':              
				#if root == None:
				#if root != self.theMainWindow.theEntityListWindow:
				if root != None and root.__class__.__name__ == 'EntityListWindow':
					self.thePropertyWindowOnEntityListWindows[ anInstance ] = None
				else:
					anInstance.editTitle( aTitle )
					self.thePluginTitleDict[ anInstance ] = aTitle
					self.theInstanceList.append( anInstance )
				# initializes session
				self.theMainWindow.theSession.theSimulator.initialize()
				self.updateFundamentalWindows()
			#except:
			#	pass

			return anInstance

	# end of createInstance

	# ========================================================================
	def loadModule( self, aClassname ):
		"""loads plugin window
		aClassname   ---   a class name of PluginWindow
		"""

		PluginManager.loadModule(self,aClassname)

	# ========================================================================
	def loadAll( self ):
		"""loads all plugin windows' files
		Returns None
		"""

		for aPath in PLUGIN_PATH:
			aFileList = glob.glob( os.path.join( aPath, '*.glade' ) )
			for aFile in aFileList:
				aModulePath = os.path.splitext( aFile )[0]
				if( os.path.isfile( aModulePath + '.py' ) ):
					aModuleName = os.path.basename( aModulePath )
					self.loadModule( aModuleName )


	# ========================================================================
	def updateAllPluginWindow( self ):
		"""updates all plugin windows
		Returns None
		"""

		# updates all plugin windows
		PluginManager.updateAllPluginWindow(self)

		# updates PropertyWindow on EntityListWindow
		if self.thePropertyWindowOnEntityListWindows != None:
			for aPropertyWindowOnEntityListWindow in self.thePropertyWindowOnEntityListWindows.keys():
				aPropertyWindowOnEntityListWindow.update()


	# ---------------------------------------------------------------
	# appendInstance  (overrides PluginManager)
	#   - appends an instance to instance list
	#   - If catchs exception from the method of super class,
	#     then print message to MessageWindow.
	#
	# anInstance     : an instance
	# return -> None
	# ---------------------------------------------------------------
	# ========================================================================
	def appendInstance( self, anInstance ):

		pass
		#print ""
		#print " anInstance = "
		#print anInstance 

		#try:
#		PluginManager.appendInstance(self, anInstance)
		#except:
		#	aMessage  = '\n----------Error------------\n'
		#	aMessage += 'ErroType[%s]\n'  %sys.exc_type
		#	aMessage += 'ErroValue[%s]\n' %sys.exc_value
		#	traceback.print_exc()
		#	self.printMessage(aMessage)

		#self.theMainWindow.update()

	# end of appendInstance


	# ========================================================================
	def removeInstance( self, anInstance ):
		"""override superclass's method
		anInstance   --- a PluginWindow instance 
		Returns None
		"""

		# calls superclass's method
		PluginManager.removeInstance(self, anInstance)

		# deletes it from the instance map
		if self.thePluginTitleDict.has_key( anInstance ):
			del self.thePluginTitleDict[ anInstance ] 
		else:
			pass

		# The following process is verbose
		# when the instance is not deleted, destroy it.
		if anInstance != None:
			if anInstance[anInstance.__class__.__name__] != None:
				anInstance[anInstance.__class__.__name__].destroy()

		# updaets fundamental windows
		self.theMainWindow.updateFundamentalWindows()


	# ========================================================================
	def removeInstanceByTitle( self, aTitle ):
		"""removes PluginWindow instance by title
		aTitle   --- a PluginWindow's title (str)
		Returns None
		"""
		
		# converts the title to str type
		aTitle = str(aTitle)

		# removes the instance
		for anInstance in self.theInstanceList:
			if aTitle == self.thePluginTitleDict[ anInstance ]:
				self.removeInstance( anInstance )
				break

	# ========================================================================
	def editModuleTitle( self, aPluginInstance, aTitle ):
		"""overwrites superclass's method
		edits PluginWindow's title
		aPluginInstance   --- the PluginWindow to change title (PluginWindow)
		aTitle            --- a new PluginWindow's title (str)
		Returns None
		"""

		self.thePluginTitleDict[aPluginInstance] = aTitle
		PluginManager.editModuleTitle( self, aPluginInstance, aTitle)

	# ========================================================================
	def editInstanceTitle( self, anOldTitle, aNewTitle ):
		"""edits PluginWindow's title
		anOldTitle   --- current PluginWindow's title (str)
		anNewTitle   --- a new PluginWindow's title (str)
		Returns None
		"""

		# converts the title to str type
		anOldTitle = str(anOldTitle)
		aNewTitle = str(aNewTitle)

		# edits the instance's title
		for anInstance in self.theInstanceList:
			#print self.thePluginTitleDict[ anInstance ]
			if anOldTitle == self.thePluginTitleDict[ anInstance ]:
				self.editModuleTitle( anInstance, aNewTitle )
				break


	# ========================================================================
	def showPlugin( self, aPluginInstance ):
		"""overwrites superclass's method
		aPluginInstance   ---  a PluginWindow instance 
		Returns None
		"""

		PluginManager.showPlugin(self, aPluginInstance)


	# ========================================================================
	def deleteModule( self, *arg ):
		"""overwrites superclass's method
		aPluginInstance   ---  a PluginWindow instance 
		Returns None
		"""

		self.theMainWindow.update()


	# ========================================================================
	def updateFundamentalWindows( self ):
		"""updates fundamental windows
		Returns None
		"""

		try:
			self.theMainWindow.updateFundamentalWindows()
			self.theMainWindow.update()
		except:
			pass


	# ========================================================================
	def printMessage( self, aMessage ):
		"""prints message on MessageWindow
		Returns None
		"""

		self.theMainWindow.printMessage(aMessage)


	# ========================================================================
	def updateAllWindows( self ):
		"""updates all windows
		Returns None
		"""

		self.updateAllPluginWindow()
		self.theMainWindow.update()
		self.theMainWindow.updateFundamentalWindows()


	# ========================================================================
	def deletePropertyWindowOnEntityListWinsow( self, aPropertyWindowOnEntityListWindow ):
		"""deletes PropertyWindow on EntityListWindow
		Returns None
		"""

		if self.thePropertyWindowOnEntityListWindows.has_key(aPropertyWindowOnEntityListWindow):
			del self.thePropertyWindowOnEntityListWindows[aPropertyWindowOnEntityListWindow]

if __name__ == "__main__":
    pass





