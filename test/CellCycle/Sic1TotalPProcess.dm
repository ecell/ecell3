CLASSNAME = 'Sic1TotalPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  VariableReference P0;
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  
'''

defineMethod( 'initialize', '''
  P0 = getVariableReference( "P0" );
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  C2 = getVariableReference( "C2" );
''' )

defineMethod( 'process', '''
  const Real E0( C0.getVariable()->getConcentration() ); // const?
  const Real E1( C1.getVariable()->getConcentration() ); // const?
  const Real E2( C2.getVariable()->getConcentration() ); // const?

  Real T( E0 - E1 - E2 );
  T *= N_A * getSuperSystem()->getVolume();
  P0.getVariable()->setValue( T );
''' )
