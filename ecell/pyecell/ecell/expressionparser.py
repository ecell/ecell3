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
# -----------------------------------------------------------------------------
#  expressionparser.py
#
#  An expression parser for SBML Exporter
#  This program is part of E-Cell Simulation Environment Version 3.
#
#  Author : Tatsuya Ishida
# -----------------------------------------------------------------------------

__program__ = 'expressionparser'
__author__ = 'Tatsuya Ishida'
__copyright__ = 'Copyright (C) 2002-2009 Keio University'
__license__ = 'GPL'

LEXTAB = "ecell.expressionlextab"
PARSERTAB = "ecell.expressionparsetab"


import os
import types

# import the lex and yacc
import ply.lex as lex
import ply.yacc as yacc

#import libsbml


tokens = (
    'COMMA','COLON','NAME','NUMBER',
    'PLUS','MINUS','TIMES','DIVIDE','POWER',
    'LPAREN','RPAREN',
    )


# Tokens

t_PLUS    = r'\+'
t_MINUS   = r'\-'
t_TIMES   = r'\*'
t_DIVIDE  = r'\/'
t_POWER   = r'\^'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'
t_NAME    = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_COLON   = r','
t_COMMA   = r'.'


def t_NUMBER(t):
    r' (\d+(\.\d*)?|\d*\.\d+)([eE][+-]?\d+)? '
        
    return t


# Ignored characters
t_ignore = " \t"

def t_newline(t):
    r'\n+'
    t.lineno += t.value.count("\n")
    
def t_error(t):
    print "Illegal character '%s'" % t.value[0]
    t.skip(1)
    

# Parsing rules

precedence = (
    ('left','PLUS','MINUS'),
    ('left','TIMES','DIVIDE'),
    ('left','POWER'),
    ('right','UMINUS'),
    )

# dictionary of names
names = { }


def p_statement_expr(t):
    'statement : expression'
    t[0] = str( t[1] )

def p_expression_binop(t):
    '''expression : expression PLUS expression
                  | expression MINUS expression
                  | expression TIMES expression
                  | expression DIVIDE expression
                  | expression POWER expression'''
#    print "binop"
    t[0] = str( t[1] + t[2] + t[3] )

def p_expression_uminus(t):
    'expression : MINUS expression %prec UMINUS'
#    print "uminus"
    t[0] = str( t[1] + t[2] )

def p_expression_group(t):
    'expression : LPAREN expression RPAREN'
#    print "group"
    t[0] = str( t[1] + t[2] + t[3] )

def p_expression_factor(t):
    '''expression : System_Function
                  | VariableReference
                  | Function
                  | NUMBER
                  | NAME '''
    t[0] = str( t[1] )


def p_expression_arguments(t):
    '''arguments : arguments COLON expression
                 | expression
                 | empty'''
#    print "arguments"
    if len(t.slice) == 4: 
        t[0] = str( t[1] + t[2] + t[3] )
    else:
        t[0] = str( t[1] )

def p_expression_system_function(t):
    'System_Function : NAME COMMA Function COMMA NAME'
#    print "system_function"

    if ( t[3] == 'getSuperSystem()' ):

        if ( t[1] == 'self' ):

            aLastSlash = aReactionPath.rindex( '/' )
            aCompartmentID = aReactionPath[aLastSlash+1:]

            if ( t[5] == 'Size' ):

                t[0] = aCompartmentID

            elif ( t[5] == 'SizeN_A' ):

                t[0] = '(' + aCompartmentID + '*N_A)'

            else:
                raise AttributeError, "getSuperSystem attribute must be Size or SizeN_A"

        else:

            for aVariableReference in aVariableReferenceList:

                if ( aVariableReference[0] == t[1] ):

                    aFastColon = aVariableReference[1].index( ':' )
                    aLastColon = aVariableReference[1].rindex( ':' )
                    
                    aSystemPath = aVariableReference[1][aFastColon+1:aLastColon]

                    if ( aSystemPath == '.' ):

                        aLastSlash = aReactionPath.rindex( '/' )
                        aCompartmentID = aReactionPath[aLastSlash+1:]

                        if ( t[5] == 'Size' ):

                            t[0] = aCompartmentID

                        elif ( t[5] == 'SizeN_A' ):

                            t[0] = '(' + aCompartmentID + '*N_A)'

                        else:
                            raise AttributeError, "getSuperSystem attribute must be Size or SizeN_A"

                    else:

                        aLastSlash = aSystemPath.rindex( '/' )
                        aCompartmentID = aSystemPath[aLastSlash+1:]

                        if ( t[5] == 'Size' ):

                            t[0] = aCompartmentID

                        elif ( t[5] == 'SizeN_A' ):

                            t[0] = '(' + aCompartmentID + '*N_A)'

                        else:
                            raise AttributeError,"getSuperSystem attribute must be Size or SizeN_A"

    else:
        raise TypeError, str( t[1] ) + " doesn't have " + str( t[3] )

            
def p_expression_function(t):
    'Function : NAME LPAREN arguments RPAREN'

    global aDelayFlag
    
    if ( t[1] == 'delay' ):
        aDelayFlag = True
        
    t[0] = str( t[1] + t[2] + t[3] + t[4] )

    
def p_expression_variablereference(t):
    'VariableReference : NAME COMMA NAME'
#    print "VariableReference"

    aVariableID = []
    for aVariableReference in aVariableReferenceList:

        if( aVariableReference[0] == t[1] ):

            aFastColon = aVariableReference[1].index( ':' )
            aLastColon = aVariableReference[1].rindex( ':' )
            
            aSystemPath = aVariableReference[1][aFastColon+1:aLastColon]
            aVariable = aVariableReference[1][aLastColon+1:]            


            # --------------------------------------------------------------
            # If there are some VariableReference which call getValue()
            # function "ID.Value", this VariableReference must be 
            # distingish between [Species] and [Parameter].
            #
            # If it is the [Species], it must be converted into
            # MolarConcentration.  "ID.Value / ( SIZE * N_A )"
            #
            # In case of [Parameter], it must be without change.
            # --------------------------------------------------------------
            
            # VariableReference attribute is Value
            if ( aSystemPath == '/SBMLParameter' ):

                if( ( ( 'SBMLParameter__' + aVariable )\
                      in theID_Namespace ) == False ):
                        
                    aVariableID.append( aVariable )
                else:
                    aVariableID.append( 'SBMLParameter__' + aVariable )

            elif ( aSystemPath == '.' and
                   aReactionPath == '/SBMLParameter' ):

                if( ( ( 'SBMLParameter__' + aVariable )\
                      in theID_Namespace ) == False ):
                        
                    aVariableID.append( aVariable )
                else:
                    aVariableID.append( 'SBMLParameter__' + aVariable )

            else:
                if ( aSystemPath == '.' ):
                    if( aReactionPath == '/' ):
                        aVariableID.append( getVariableID
                                            ( aVariable, '/', t[3] ) )
                    else:
                        aVariableID.append( getVariableID( aVariable,\
                                                           aReactionPath[1:],\
                                                           t[3] ) )
                else:
                    if( aSystemPath == '/' ):
                        aVariableID.append( getVariableID
                                            ( aVariable, '/', t[3] ) )
                    else:
                        aVariableID.append( getVariableID( aVariable,\
                                                           aSystemPath[1:],\
                                                           t[3] ) )
                        
    if ( aVariableID == [] ):
        raise NameError, "not find VariableReference ID"

    t[0] = aVariableID[0]



def p_empty(t):
    '''
    empty :
    '''
#        print "Empty"
    t[0] = ''

def p_error(t):
    print "Syntax error at '%s'" % t.value




def initializePLY(outputdir):
    lextabmod = LEXTAB.split('.')
    parsertabmod = PARSERTAB.split('.')
    lex.lex( lextab=lextabmod[-1], optimize=1, outputdir=os.path.join( outputdir,*lextabmod[:-1] ) )
    yacc.yacc( tabmodule=parsertabmod[-1], outputdir=os.path.join( outputdir, *parsertabmod[:-1] ) )


def isID_Namespace( aVariableID ):

    return ( aVariableID in theID_Namespace )


def getVariableID( aVariableID, aPath, aType ):

    if( aPath.count('/') == 0 ):
        aSystem = aPath
    else:
        aLastSlash = aPath.rindex( '/' )
        aSystem = aPath[aLastSlash+1:]

    # in case of Compartment
    if( aVariableID == 'SIZE' ): 
        
        if ( aPath == '' ): # Root system
            return 'default'

        else: # other system
            if( isID_Namespace( 'default__' +\
                                aPath.replace( '/', '__' ) ) ):

                return 'default__' + aPath.replace( '/', '__' )
            else:                                    
                return aSystem

    # in case of Species
    else:
        if ( aPath == '' ): # Root system
            
            if( isID_Namespace( 'default__' + aVariableID ) ):
                
                aVariableID = 'default__' + aVariableID

            aSystem = 'default'
            
        else: # other system
            if( isID_Namespace( aPath.replace( '/', '__' ) + '__' +\
                                aVariableID ) ):
                
                aVariableID = aPath.replace( '/', '__' ) + '__' +\
                              aVariableID


    if( aType == 'Value' ):

        return '(' + aVariableID + '/' + aSystem + '/N_A)'
        
    elif( aType == 'NumberConc' ):

        return '(' + aVariableID + '/N_A)'

    elif( aType == 'MolarConc' ):

        return aVariableID

    else:
        raise AttributeError,"VariableReference attribute must be MolarConc, NumberConc and Value"
                    


def convertExpression( anExpression, aVariableReferenceListObject, aReactionPathObject, ID_Namespace, debug=0 ):
  
    global aVariableReferenceList
    global aReactionPath
    global aDelayFlag
    global theID_Namespace
    
    aVariableReferenceList = aVariableReferenceListObject
    aReactionPath = aReactionPathObject
    aDelayFlag = False
    theID_Namespace = ID_Namespace
    
    aLexer = lex.lex( lextab="expressionlextab" )    
    aParser = yacc.yacc( optimize=1, tabmodule="expressionparsetab" )

    return [ aParser.parse( anExpression, lexer=aLexer, debug=debug ),
             aDelayFlag ]
