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
#
#'Design: Koichi Takahashi <shafi@e-cell.org>',
#'Programming: Masahiro Sugimoto <sugi@bioinformatics.org>'
#
# E-Cell Project, Lab. for Bioinformatics, Keio University.
#

from ecell.ecssupport import *

# ----------------------------------------------------------------------
# Index of stepper's proprety
# ----------------------------------------------------------------------

SETTABLE = 0
GETABLE  = 1

# ----------------------------------------------------------------------
# View types of OsogoPluginWindow
# ----------------------------------------------------------------------

SINGLE   = 0
MULTIPLE = 1

# ----------------------------------------------------------------------
# decodeAttribute
#
# anAttribute : an attribute ( TRUE or FALSE )
#
# return -> '+' or '-'
# ----------------------------------------------------------------------
def decodeAttribute(anAttribute):

	if anAttribute == TRUE:
		return '+'
	else:
		return '-'

# end of decodeAttribute


# ----------------------------------------------------------------------
# convertStringToTuple
#
# aString : a string as below
#           "(a,b,c)"
#
# return -> '+' or '-'
# ----------------------------------------------------------------------
def convertStringToTuple(aString):

	aString = aString[1:-1]
	aList = aString.split(',')

	for anIndex in range(0,len(aList)):
		anElement = aList[anIndex]
		anElement = anElement.strip()
		try:
			anElement = int(anElement)
		except:
			try:
				anElement = float(anElement)
			except:
				anElement = anElement[1:-1]

		aList[anIndex] = anElement

	return tuple(aList)

# end of convertStringToTuple


# ----------------------------------------------------------------------
# shortenString
#
# aValue : an original string
# aNumber : the length to cut original string
#
# return -> a shorten string
# ----------------------------------------------------------------------
def shortenString( aValue, aNumber):

	if len( str(aValue) ) > aNumber:
		return aValue[:aNumber] + '...'
	else:
		return aValue
		
# end of shortenString


