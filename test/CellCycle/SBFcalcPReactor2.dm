CLASSNAME = 'SBFcalcPReactor2'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','k3',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','J1',0.0),('Real','J2',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant C1;
  Reactant C2;
  Reactant C3;
  Reactant C4;
  Reactant P0;
'''


defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  C2 = getReactant( "C2" );
  C3 = getReactant( "C3" );
  C4 = getReactant( "C4" );
  P0 = getReactant( "P0" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getSubstance()->getConcentration() ); // const
  const Real E2( C1.getSubstance()->getConcentration() ); // const
  const Real E3( C2.getSubstance()->getConcentration() ); // const
  const Real E4( C3.getSubstance()->getConcentration() ); // const
  const Real E5( C4.getSubstance()->getConcentration() ); // const
  Real Va( E1 + e1 * (E2 + E3) + e2 * E4);
  Va *= k3;
  Real Vi( k1 + k2 * E5);

  Real a( Vi - Va );
  Real b( Vi - Va + Va * J2 + Vi * J1 );
  Real g( Va * J2 );

  Real F( 2 * g );
  F /= b + sqrt( b*b - 4*a*g );
  F *= getSuperSystem()->getVolume() * N_A;

  P0.getSubstance()->setQuantity( F );
''' )
