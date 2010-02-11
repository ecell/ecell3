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
#
# written by Masahiro Sugimoto <sugi@bioinformatics.org> at
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

# ------------------------------------------------------------------
# LoggerFactory print out a data to file.
# Some data format can be created and printet out.
# The formats are ECD and binary.
# ------------------------------------------------------------------
class LoggerFactory:

	def __init__(self):
		self.theECDExtension='ecd'

	# ------------------------------------------------------------------
	# print out ECD format data to file.
	#
	# aDataName(string) : the name of Data which is one item of ECD format
	# aMatrixData(touple of touple of float)
	#					    : matrix data (dimension is n(columns) * m(lines) )
	# aDataName(string) : the dataname which is one item of ECD format
	# aLabel(string)    : the label which is one item of ECD format
	# aNote(string)     : the note which is one item of ECD format
	#
	# return  -> ECDFormatData
	#
	# This method is thorwable exception.
	# ------------------------------------------------------------------
	def printECDFormatDataToFile( self, aFileName, aMatrixData, 
								aDataName='', aLabel='',
								aNote=''):

		##print "1aFileName = %s" %aFileName
		##print 'len = %d' %(len(self.theECDExtension))

		if len(aFileName) == 0:
			raise " Error(%s) : The length of file name is 0 !" %__name__

		if aFileName.rfind( self.theECDExtension ) == \
			len(aFileName)-len(self.theECDExtension):
			pass
		else:
			aFileName = aFileName + self.theECDExtension

		##print "2aFileName = %s" %aFileName

		aOutputFile= open(aFileName,"w")

		# print out DATA
		aOutputFile.writelines("DATA: %s\n" %aDataName)

		# check the size of matrix
		theColumnNumber=0
		if(len(aMatrixData)>1):
			theColumnNumber=len(aMatrixData[0])

		# print out SIZE
		aOutputFile.writelines("SIZE: %d %d\n" %(theColumnNumber,len(aMatrixData)))

		# print out LABEL
		aOutputFile.writelines("LABEL: %s\n" %aLabel)

		# print out NOTE
		aOutputFile.writelines("NOTE: %s\n" %aNote)

		# print out horizontal line
		aOutputFile.writelines("\n----------------------\n")

		# print out matrix
		if(theColumnNumber>=1):
			theColumnNumberMinusOne = theColumnNumber - 1
			for i in xrange(len(aMatrixData)):
				j=0
				while j < theColumnNumberMinusOne:
					aOutputFile.write("%s\t" %aMatrixData[i][j])
					j=j+1
				aOutputFile.write("%s\n" %aMatrixData[i][theColumnNumberMinusOne])

		# close the file 
		aOutputFile.close()

	# end of PrintECDFormatDataToFile

	# ------------------------------------------------------------------
	# return the file name 
	#
	# aFullPNString(string) : the name of file to print out data
	#
	# return  aFilename -> if sucseed
	#
	# This method is thorwable exception.
	# ------------------------------------------------------------------
	def getECDFileNameFromFullPNString( self, aFullPNString ):

		aFilename = aFullPNString.split('/')
		aFilename = aFilename[-1]
		aFilename = aFilename.split(':')
		aFilename = aFilename[0] + '_' + aFilename[1] + '_' + aFilename [-1]
		return aFilename + '.ecd'

	# end of theECDFileName

	# ------------------------------------------------------------------
	# return the directory 
	#
	# aFullPNString(string) : the name of file to print out data
	#
	# return  directoryname -> if sucseed
	#         None          -> if some errors were catched.
	# 
	# This method is thorwable exception.
	# ------------------------------------------------------------------
	def getDirectoryNameFromFullPNString( self, aFullPNString):
		aDirectoryname=aFullPNString.split(':')
		aDirectoryname=aDirectoryname[1]
		aDirectoryname=aDirectoryname.split('/')
		return aDirectoryname[-1]

	# end of theDirectory


# end of LoggerFactory


if __name__ == "__main__":

	def main():
		data = [[0,1],[2,3],[4,5]]
		#data = [[0],[2],[4]]
		a = LoggerFactory()
		#status = a.printECDFormatDataToFile('hoge',data,'dammy','no','aa')
		status = a.printECDFormatDataToFile('hoge',data,'dammy','no')
		print status

	main()
