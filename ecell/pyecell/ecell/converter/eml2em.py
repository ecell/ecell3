#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2007 Keio University
#       Copyright (C) 2005-2007 The Molecular Sciences Institute
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

__program__ = 'ecell3-eml2em'
__version__ = '0.1'
__author__ = 'Kentarou Takahashi and Koichi Takahashi <shafi@e-cell.org>'
__copyright__ = 'Copyright (C) 2002-2004 Keio University'
__license__ = 'GPL'

__all__ = (
    'convert'
    )

import re
import types
import string

from ecell.ecssupport import *
from ecell.eml import *

# taken from emllib.
class EMRenderer:
    numberPattern = r'^[+-]?(\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)?$'
    unquotedStringPattern = r'^[a-zA-Z_/\:][\w\:/\.]*$'
    quoteRe = re.compile( numberPattern + '|' + unquotedStringPattern )

    def __init__( self, aBuffer ):
        self.theBuffer = aBuffer

    def expandValueList( aValue, aLevel=0 ):
        aType = type( aValue )

        # Three cases: (1) string, (2) other non-list types, and
        #              (3) list (or tuple).

        # (1) string.  Quote it if necessary.
        if aType == str or aType == unicode:
            if self.quoteRe.match( aValue ) is None:
                return '"' + aValue + '"'
            else:
                return aValue

        # (2) other non-list types (such as numbers).  Simply stringify.
        if aType != types.TupleType and aType != types.ListType:
            return str( aValue )

        # (3) list or tuple.
        #     Recursively apply this method for nested lists.
        
        aValue = map( expandValueList, aValue, ( aLevel+1, ) * len( aValue ) )

        aValueString = ''

        # don't print a space before the first item
        if len( aValue ) > 0:
            aValueString += str( aValue[0] )
            
        for anItem in aValue[1:]:
            aValueString += ' '
            aValueString += str( anItem )

        # don't use [] in uppermost level
        if aLevel != 0:
            aValueString = '[ ' + aValueString + ' ]'

        return aValueString
    expandValueList = staticmethod( expandValueList )


    def renderNewline( self ):
        self.theBuffer += "\n"

    def renderObjectDef( self, aType, aClass, anArgList, aPropertyList, anOptional='' ):
        # declaration line
        anArgListString = string.join( anArgList, ',' ) 
        self.theBuffer += '%s %s( %s )\n{\n' % ( aType, aClass, anArgListString )

        # properties
        if len( aPropertyList ) != 0:
            for aProperty in aPropertyList:
                aPropertyName = aProperty[0]
                aPropertyValueListString = expandValueList( aProperty[1] )
                self.theBuffer += '\t%s\t%s;\n' % ( aPropertyName,
                                                    aPropertyValueListString )
    #            print aPropertyName
        else:
            self.theBuffer += '\t# no property\n'
        
        # optional field
        if anOptional != '':
            self.theBuffer += '\n\t'
            self.theBuffer += string.replace( anOptional, '\n', '\n\t' )
            self.theBuffer += '\n'

        self.theBuffer += '}\n'

    def renderStepperList( self, anEml ):
        for aStepperID in anEml.getStepperList():
            aType = 'Stepper'
            aClass = anEml.getStepperClass( aStepperID )
            anArgList = ( aStepperID, )
            aPropertyNameList = anEml.getStepperPropertyList( aStepperID )
            aPropertyList = []
            for aPropertyName in aPropertyNameList:
                aPropertyValue = anEml.getStepperProperty( aStepperID,
                                                           aPropertyName )
                aPropertyList.append( ( aPropertyName, aPropertyValue ) )
            self.renderObjectDef( 'Stepper', aClass, anArgList, aPropertyList )
            self.renderNewline()

    def renderEntity( self, anEml, aFullID, anOptional='' ):
        aFullIDString = renderFullIDString( aFullID )
        aType = ENTITYTYPE_STRING_LIST[ aFullID[ TYPE ] ]
        aClass = anEml.getEntityClass( aFullIDString )

        if aFullID[TYPE] != SYSTEM:
            anArgList = ( aFullID[ ID ], )
        else:
            if len( aFullID[SYSTEMPATH] ) == 0 or aFullID[SYSTEMPATH][-1] == '/':
                aSystemPath = aFullID[SYSTEMPATH] + aFullID[ID]
            else:
                aSystemPath = aFullID[SYSTEMPATH] + '/' + aFullID[ID]
            anArgList = ( aSystemPath, )

        aPropertyNameList = anEml.getEntityPropertyList( aFullIDString )
        aPropertyList = []
        for aPropertyName in aPropertyNameList:
            aFullPN = aFullIDString + ':' + aPropertyName
            aPropertyValue = anEml.getEntityProperty( aFullPN )
            aPropertyList.append( ( aPropertyName, aPropertyValue ) )

        self.renderObjectDef( aType, aClass, anArgList, aPropertyList, anOptional )
        
    def renderSystemList( self, anEml, aSystemPath='/' ):
        for anID in anEml.getEntityList( 'Variable', aSystemPath ):
            aFullID = ( VARIABLE, aSystemPath, anID )
            self.renderEntity( anEml, aFullID )
            self.renderNewline()

        for anID in anEml.getEntityList( 'Process', aSystemPath ):
            aFullID = ( PROCESS, aSystemPath, anID )
            anOptional += self.renderEntity( anEml, aFullID )
            self.renderNewline()

        if aSystemPath == '':
            aFullID = ( SYSTEM, '', '/' )
        else:
            aLastSlash = string.rindex( aSystemPath, '/' )
            aPath = aSystemPath[:aLastSlash+1]
            anID = aSystemPath[aLastSlash+1:]
            aFullID = ( SYSTEM, aPath, anID )

        aBuffer += self.renderEntity( anEml, aFullID, anOptional )
        aBuffer += '\n'

        for aSystem in anEml.getEntityList( 'System', aSystemPath ):
            aSubSystemPath = joinSystemPath( aSystemPath, aSystem )
            aBuffer += self.renderSystemList( anEml, aSubSystemPath )

        return aBuffer

def convert( anEmlDocument ):
    anEml = Eml( anEmlDocument )
    aBuffer = ''
    aBuffer += renderStepperList( anEml )
    aBuffer += renderSystemList( anEml )

    return aBuffer
