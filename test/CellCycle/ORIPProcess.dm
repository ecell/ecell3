CLASSNAME = 'ORIPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference C1;
  VariableReference P0;
  Real prev; // declare member variable
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  P0 = getVariableReference( "P0" );
  prev = 0; // initialize member variable
''' )

defineMethod( 'process', '''
  // Real prev( 0 ); // removed static

  const Real E1( C0.getVariable()->getConcentration() ); // const?
  const Real E2( C1.getVariable()->getConcentration() ); // const?

  Real E( E1 + E2 );

  if(prev > 0.2 && E <= 0.2)
  {
    P0.getVariable()->setValue( 0 );
  }

  prev = E;
''' )
