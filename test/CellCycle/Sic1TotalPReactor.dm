CLASSNAME = 'Sic1TotalPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = []

PROTECTED_AUX = '''
  Reactant P0;
  Reactant C0;
  Reactant C1;
  Reactant C2;
  
'''

defineMethod( 'initialize', '''
  P0 = getReactant( "P0" );
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  C2 = getReactant( "C2" );
''' )

defineMethod( 'react', '''
  const Real E0( C0.getSubstance()->getConcentration() ); // const?
  const Real E1( C1.getSubstance()->getConcentration() ); // const?
  const Real E2( C2.getSubstance()->getConcentration() ); // const?

  Real T( E0 - E1 - E2 );
  T *= N_A * getSuperSystem()->getVolume();
  P0.getSubstance()->setQuantity( T );
''' )
