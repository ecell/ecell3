
CLASSNAME = 'Hct1InactivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  VariableReference C3;
'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  C2 = getVariableReference( "C2" );
  C3 = getVariableReference( "C3" );
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

  setFlux( V );
''' )

