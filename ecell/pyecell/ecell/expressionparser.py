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
__copyright__ = 'Copyright (C) 2002-2004 Keio University'
__license__ = 'GPL'


import string
import types

# import the lex and yacc
import lex
import yacc

import libsbml


tokens = (
    'COMMA','COLON','NAME','NUMBER',
    'PLUS','MINUS','TIMES','DIVIDE',
    'LPAREN','RPAREN',
    )


# Tokens

t_PLUS    = r'\+'
t_MINUS   = r'\-'
t_TIMES   = r'\*'
t_DIVIDE  = r'\/'
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
                  | expression DIVIDE expression'''
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

            aLastSlash = string.rindex( aReactionPath, '/' )
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

                    aFastColon = string.index( aVariableReference[1], ':' )
                    aLastColon = string.rindex( aVariableReference[1], ':' )
                    
                    aSystemPath = aVariableReference[1][aFastColon+1:aLastColon]

                    if ( aSystemPath == '.' ):

                        aLastSlash = string.rindex( aReactionPath, '/' )
                        aCompartmentID = aReactionPath[aLastSlash+1:]

                        if ( t[5] == 'Size' ):

                            t[0] = aCompartmentID

                        elif ( t[5] == 'SizeN_A' ):

                            t[0] = '(' + aCompartmentID + '*N_A)'

                        else:
                            raise AttributeError, "getSuperSystem attribute must be Size or SizeN_A"

                    else:

                        aLastSlash = string.rindex( aSystemPath, '/' )
                        aCompartmentID = aSystemPath[aLastSlash+1:]

                        if ( t[5] == 'Size' ):

                            t[0] = aCompartmentID

                        elif ( t[5] == 'SizeN_A' ):

                            t[0] = '(' + aCompartmentID + '*N_A)'

                        else:
                            raise AttributeError, "getSuperSystem attribute must be Size or SizeN_A"

    else:
        raise TypeError, str( t[1] ) + " doesn't have " + str( t[3] )

    		
def p_expression_function(t):
    'Function : NAME LPAREN arguments RPAREN'
#    print "function"
    t[0] = str( t[1] + t[2] + t[3] + t[4] )

    
def p_expression_variablereference(t):
    'VariableReference : NAME COMMA NAME'
#    print "VariableReference"

    aVariableID = []
    for aVariableReference in aVariableReferenceList:

        if( aVariableReference[0] == t[1] ):

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
            
            if ( t[3] == 'Value' ):

                aFastColon = string.index( aVariableReference[1], ':' )
                aLastColon = string.rindex( aVariableReference[1], ':' )

                aSystemPath = aVariableReference[1][aFastColon+1:aLastColon]

                if ( aSystemPath == '/SBMLParameter' ):

                    aVariableID.append( getVariableReferenceId( aVariableReference[1] ) )

                elif ( aSystemPath == '.' and
                       aReactionPath == '/SBMLParameter' ):

                    aVariableID.append( getVariableReferenceId( aVariableReference[1] ) )

                else:
                    aVariableID.append( getVariableReferenceId( aVariableReference[1] ) )
                    if ( aSystemPath == '.' ):

                        aLastSlash = string.rindex( aReactionPath, '/' )
                        aVariableID[0] = '(' + aVariableID[0]+ '/'+ aReactionPath[aLastSlash+1:] + '/' + 'N_A)'

                    else:
                        aLastSlash = string.rindex( aSystemPath, '/' )
                        aVariableID[0] = '(' + aVariableID[0] + '/'+ aSystemPath[aLastSlash+1:] + '/' + 'N_A)'

            elif ( t[3] == 'NumberConc' ):
                
                aVariableID.append( getVariableReferenceId( aVariableReference[1] ) )
                aVariableID[0] = '(' + aVariableID[0] + '/N_A)'

            elif ( t[3] == 'MolarConc' ):

                aVariableID.append( getVariableReferenceId( aVariableReference[1] ) )

            else:
                raise AttributeError, "VariableReference attribute must be MolarConc, NumberConc and Value"
                    
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




def initializePLY():
    lextabname = "expressionlextab"
    yacctabname = "expressionparsetab"

    lex.lex( lextab=lextabname, optimize=1 )
#    lex.lex()

    return yacc.yacc( optimize=1, tabmodule=yacctabname )
#    return yacc.yacc()



# -------------------------------
# return the VariableReference ID
# -------------------------------

def getVariableReferenceId( aVariableReference ):

    aFastColon = string.index( aVariableReference, ':' )
    aLastColon = string.rindex( aVariableReference, ':' )

    # set Species Id to Reactant object
    if ( aVariableReference[aFastColon+1:aLastColon] == '.' ):

        aSpeciesReferencePath = string.replace( aReactionPath[1:], '/', '_S_' )
        
    else:
        aSpeciesReferencePath = string.replace( aVariableReference[aFastColon+2:aLastColon], '/', '_S_' )

    return  aSpeciesReferencePath + '_' + aVariableReference[aLastColon+1:]



def convertExpression( anExpression, aVariableReferenceListObject, aReactionPathObject ):

     global aVariableReferenceList
     global aReactionPath
     aVariableReferenceList = aVariableReferenceListObject
     aReactionPath = aReactionPathObject

     aYacc = initializePLY()

     return aYacc.parse( anExpression )
