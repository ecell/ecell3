
CLASSNAME = 'Hct1ActivateReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Reactant P0;
  Reactant C0;
  Reactant C1;
'''

defineMethod( 'initialize', '''
  P0 = getReactant( "P0" );
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
''' )

defineMethod( 'react', '''
  const Real P( P0.getSubstance()->getConcentration() ); 
  const Real E1( C0.getSubstance()->getConcentration() ); 
  const Real E2( C1.getSubstance()->getConcentration() ); 
  Real V( k1 + k2 * E1 );
  V *= E2 - P;
  V /= J + E2 - P;
  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

