CLASSNAME = 'ORIPReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = []

PROTECTED_AUX = '''
  Reactant C0;
  Reactant C1;
  Reactant P0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
  P0 = getReactant( "P0" );
''' )

defineMethod( 'react', '''
  static Real prev( 0 );

  const Real E1( C0.getSubstance()->getConcentration() ); // const?
  const Real E2( C1.getSubstance()->getConcentration() ); // const?

  Real E( E1 + E2 );

  if(prev > 0.2 && E <= 0.2)
  {
    P0.getSubstance()->setQuantity( 0 );
  }

  prev = E;
''' )
