CLASSNAME = 'Sic1TotalPProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = []

PROTECTED_AUX = '''
  Connection P0;
  Connection C0;
  Connection C1;
  Connection C2;
  
'''

defineMethod( 'initialize', '''
  P0 = getConnection( "P0" );
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  C2 = getConnection( "C2" );
''' )

defineMethod( 'react', '''
  const Real E0( C0.getVariable()->getConcentration() ); // const?
  const Real E1( C1.getVariable()->getConcentration() ); // const?
  const Real E2( C2.getVariable()->getConcentration() ); // const?

  Real T( E0 - E1 - E2 );
  T *= N_A * getSuperSystem()->getVolume();
  P0.getVariable()->setValue( T );
''' )
