#!/usr/bin/python
#
# ZipManager.py  - to start up E-Cell3 Tool Launcher
#

from config import *
from shutil import *
from fileFind import *
from Preferences import *

import os
import os.path
import sys
import re
import zipfile

MSG_MODELEXISTS         = "\" already exists.\nPlease enter the appropriate name."
MSG_SUCCESSFUL          = "The model \""
MSG_SUCCESSFUL_IMPORT   = "\" has been imported. "
MSG_SUCCESSFUL_EXPORT   = "\" has been exported. "

MSG_IMPUT_NAME          = "Please enter the appropriate name."
MSG_EXISTS_FILE_NAME    = "\" does not exist. Please enter the appropriate name of the file."
MSG_EXISTS_PATH_NAME    = "\" does not exist. Please enter the appropriate name of the folder."

class ZipManager:

	# ==========================================================================
	# Constractor
	# ==========================================================================
	def __init__( self, aToolLauncher ):
		"""Constructor 
		"""
		self.theToolLauncher = aToolLauncher
	# end of __init__


	# ==========================================================================
	def compress( self, exportPath ):

		exportPath, chekFile = self.checkImportFileName( exportPath )
		if chekFile == 1:
			self.createZip( exportPath )
			msg = MSG_SUCCESSFUL + exportPath + MSG_SUCCESSFUL_EXPORT
			self.printMessage( msg, msg )

	# end of compress


	# ==========================================================================
	def uncompress( self, importFileName ):

		importFileName, chekFile = self.checkImportFileName( importFileName )
		if chekFile == 1:
			if self.checkRepeat( importFileName ) == 1:
				self.openZip( importFileName )
				msg = MSG_SUCCESSFUL + importFileName + MSG_SUCCESSFUL_IMPORT
				self.printMessage( msg, msg )

		path, name = os.path.split( importFileName )
		return name[0:-4]

	# end of uncompress


	# ==========================================================================
	def openZip( self, zfileName ):

		currentDir = os.getcwd()
		os.chdir( self.getModelHome() )
		zfile = zipfile.ZipFile( zfileName,'r')
		for filename in zfile.namelist():
			data = zfile.read( filename )
			( zippath, zipfilename ) = os.path.split(filename)

			if zipfilename == '':
				os.makedirs( filename )
			else:
				self.createDir( filename )
				file = open(filename, 'w+b')
				file.write(data)
				file.close()
		os.chdir( currentDir )

	# end of openZip


	# ==========================================================================
	def createZip( self, exportPath ):

		currentDir = os.getcwd()
		compressaPath, compressName = self.getCompressPath( exportPath )
		os.chdir( exportPath )
		list = fileFind( )
		compressList = []

		for filename in list:
			filenameregex = re.split( '[\W]+',filename, 1)
			compressList.append( os.path.join(compressName, filenameregex[1]) )

		os.chdir( compressaPath )
		archive = zipfile.ZipFile( self.getZipFileName( compressName ), 'w', zipfile.ZIP_DEFLATED )
		for name in compressList:
			( zippath, zipfilename ) = os.path.split(name)
			if zipfilename != '':
				destination = name
				archive.write( name, destination )
		archive.close()
		os.chdir( currentDir )

	# end of createZip


	# ==========================================================================
	def createDir( self, fileName ):
		( path, name ) = os.path.split(fileName)
		if os.path.isdir( path ) == False:
			os.makedirs( path )
	# end of createDir


	# ==========================================================================
	def getModelHome( self ):

		pref = Preferences( self.theToolLauncher )
		return pref.getModelHome()

	# end of getModelPath


	# ==========================================================================
	def printMessage( self, windowMessage, boxMessage ):

		if windowMessage != "":
			self.theToolLauncher.printMessage( windowMessage )

		if boxMessage != "":
			self.theToolLauncher.viewErrorMessage( boxMessage )

	# end of printMessage


	# ==========================================================================
	def checkExportFileName( self, exportPath ):

		returnFlg = 1
		pathName = exportPath
		if exportPath == "":
			self.printMessage( MSG_IMPUT_NAME, MSG_IMPUT_NAME )
			returnFlg = 0
		elif os.path.exists( exportPath ) == 0:

			if os.path.exists( self.getModelHome()+os.sep+exportPath ) == 0:
				msg = "\""+exportPath+MSG_EXISTS_PATH_NAME
				self.printMessage( msg, msg )
				returnFlg = 0
			else:
				fileName = self.getModelHome()+os.sep+exportPath
		else:
			returnFlg = 1
		return exportPath, returnFlg


	# end of checkExportFileName


	# ==========================================================================
	def checkImportFileName( self, importFileName ):

		returnFlg = 1
		fileName = importFileName
		if importFileName == "":
			self.printMessage( MSG_IMPUT_NAME, MSG_IMPUT_NAME )
			returnFlg = 0
		elif os.path.exists( importFileName ) == 0:
			if os.path.exists( self.getModelHome()+os.sep+importFileName ) == 0:
				msg = "\""+importFileName+MSG_EXISTS_FILE_NAME
				self.printMessage( msg, msg )
				returnFlg = 0
			else:
				fileName = self.getModelHome()+os.sep+importFileName
		else:
			returnFlg = 1
		return fileName, returnFlg
	# end of checkImportFileName


	# ==========================================================================
	def getModelName( self, importFileName ):

		( path, fileName ) = os.path.split(importFileName)
		return re.split( '[\W]+',fileName, 1)[0]

	# end of getModelName


	# ==========================================================================
	def getCompressPath( self, exportPath ):

		( path, name ) = os.path.split( exportPath )
		return path, name

	# end of getCompressPath


	# ==========================================================================
	def getZipFileName( self, folderName ):

		modelPath = self.getModelHome()
		return modelPath+os.sep+folderName+".zip"

	# end of getZipFileName


	# ==========================================================================
	def checkRepeat( self, importFileName ):

		modelName = self.getModelName( importFileName )
		if os.path.exists( self.getModelHome()+os.sep+modelName ) == 1:
			msg = "\"" + modelName + MSG_MODELEXISTS
			self.printMessage( msg, msg )
			return 0
		else:
			return 1

	# end of checkRepeat

# end of ZipManager
