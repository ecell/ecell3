CLASSNAME = 'MAPPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference S1;
  VariableReference P0;
'''
defineMethod( 'initialize','''
  S0 = getVariableReference( "S0" );
  S1 = getVariableReference( "S1" );
  P0 = getVariableReference( "P0" );
''')

defineMethod( 'process',
'''
  const Real s0( S0.getVariable()->getValue() );
  const Real s1( S1.getVariable()->getValue() );
  
  const Real p( s0 + s1 );
  
  P0.getVariable()->setValue( p );
''')

