
CLASSNAME = 'Hct1InactivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Connection S0;
  Connection C0;
  Connection C1;
  Connection C2;
  Connection C3;
'''

defineMethod( 'initialize', '''
  S0 = getConnection( "S0" );
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  C2 = getConnection( "C2" );
  C3 = getConnection( "C3" );
''' )

defineMethod( 'process', '''
  const Real S( S0.getVariable()->getConcentration() );
  const Real E1( C0.getVariable()->getConcentration() );
  const Real E2( C1.getVariable()->getConcentration() );
  const Real E3( C2.getVariable()->getConcentration() );
  const Real E4( C3.getVariable()->getConcentration() );
  Real v( E1 + e1 * E2 + e2 * E3 + e3 * E4 );
  v *= k2;
  v += k1;
  Real V( v * S ); 
  V /= J + S;
  V += getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

