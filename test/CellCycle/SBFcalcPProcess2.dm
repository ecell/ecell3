CLASSNAME = 'SBFcalcPProcess2'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','k3',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','J1',0.0),('Real','J2',0.0)]

PROTECTED_AUX = '''
  Connection C0;
  Connection C1;
  Connection C2;
  Connection C3;
  Connection C4;
  Connection P0;
'''


defineMethod( 'initialize', '''
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  C2 = getConnection( "C2" );
  C3 = getConnection( "C3" );
  C4 = getConnection( "C4" );
  P0 = getConnection( "P0" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getVariable()->getConcentration() ); // const
  const Real E2( C1.getVariable()->getConcentration() ); // const
  const Real E3( C2.getVariable()->getConcentration() ); // const
  const Real E4( C3.getVariable()->getConcentration() ); // const
  const Real E5( C4.getVariable()->getConcentration() ); // const
  Real Va( E1 + e1 * (E2 + E3) + e2 * E4);
  Va *= k3;
  Real Vi( k1 + k2 * E5);

  Real a( Vi - Va );
  Real b( Vi - Va + Va * J2 + Vi * J1 );
  Real g( Va * J2 );

  Real F( 2 * g );
  F /= b + sqrt( b*b - 4*a*g );
  F *= getSuperSystem()->getVolume() * N_A;

  P0.getVariable()->setValue( F );
''' )
