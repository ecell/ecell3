
CLASSNAME = 'Cdc20ActivateReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k',0.0)]

PROTECTED_AUX = '''
  Reactant P0;
  Reactant C0;
'''

defineMethod( 'initialize', '''
  P0 = getReactant( "P0" );
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  const Real P( P0.getSubstance()->getConcentration());
  const Real E1( C0.getSubstance()->getConcentration());
  Real V( k * (E1 - P) );

  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )


