CLASSNAME = 'Swi5calcPProcess2'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','k3',0.0),('Real','J1',0.0),('Real','J2',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
  VariableReference P0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
  P0 = getVariableReference( "P0" );
''' )

defineMethod( 'process', '''
  const Real E1( C0.getVariable()->getConcentration() ); // const?
  const Real E2( C0.getVariable()->getConcentration() ); // const?

  Real Va( k1 * E1 );
  Real Vi( k2 + k3 * E2 );

  Real a( Vi - Va );
  Real b( a + Va * J2 + Vi * J1 );
  Real g( Va * J2 );

  Real F( 2 * g );
  F /= b + sqrt( b * b - 4 * a * g );

  F *= getSuperSystem()->getVolume() * N_A;

  P0.getVariable()->setValue( F );

''' )
