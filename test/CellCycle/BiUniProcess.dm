
CLASSNAME = 'BiUniProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX = '''
  Connection C0;
  Connection C1;
'''

defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getVariable()->getConcentration() ); 
  const Real E2( C1.getVariable()->getConcentration() ); 
  Real V( k * E1 * E2 );

  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )


