#!/usr/bin/env python2

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#        This file is part of E-Cell Session Monitor package
#
#                Copyright (C) 2001-2004 Keio University
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER
#
# Design: Kouichi Takahashi <shafi@e-cell.org>
# Design and Programming: Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from DataFile import *
from string import *

# ------------------------------------------------------------------
# DataFileManager (This is abstract class)
#   - manages some DataFile objects
# ------------------------------------------------------------------
class DataFileManager:

	# ------------------------------------------------------------------
	# Constructor
	#
	# return  -> None
	# ------------------------------------------------------------------
	def __init__(self):

		self.theFileMap = {}
		self.theRootDirectory = '.'

	# end of __init__


	# ------------------------------------------------------------------
	# getFileMap()
	#
	# return -> None
	# ------------------------------------------------------------------
	def getFileMap(self):

		return self.theFileMap

	# end of getFileMap


	# ------------------------------------------------------------------
	# setRootDirectory()
	#
	# aRootDirectory : directory to save files
	#
	# return -> None
	# ------------------------------------------------------------------
	def setRootDirectory( self, aRootDirectory ):

		self.theRootDirectory = aRootDirectory

	# end of setRootDirectory


	# ------------------------------------------------------------------
	# getRootDirectory()
	#
	# return -> aRootDirectory(string) 
	# ------------------------------------------------------------------
	def getRootDirectory( self ):

		return self.theRootDirectory 

	# end of getRootDirectory


	# ------------------------------------------------------------------
	# saveAll()
	#
	# return -> None
	# ------------------------------------------------------------------
	def saveAll(self):

		for aKey in self.theFileMap.keys():
			aFileName = self.theFileMap[aKey].getFileName()
			aFileName = split(aFileName,'/')[-1]
			aFileName = self.theRootDirectory + '/' + aFileName
			self.theFileMap[aKey].save(aFileName)

	# end of saveAll()


	# ------------------------------------------------------------------
	# loadAll()
	#
	# return -> None
	# ------------------------------------------------------------------
	def loadAll(self):

		for aKey in self.theFileMap.keys():
			self.theFileMap[aKey].load()

	# end of loadAll()


# end of DataFile

if __name__ == "__main__":

	from ECDDataFile import *

	def main():

		ecdFile = ECDDataFile()
		ecdFile.setFileName('hoge')

		dm = DataFileManager()
		dm.getFileMap()['file'] = ecdFile 
		dm.loadAll()

		dm.getFileMap()['file'].setFileName('hoge1')
		dm.saveAll()

	main()

