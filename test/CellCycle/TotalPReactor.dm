CLASSNAME = 'TotalPReactor'
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
  Real V( C0.getSubstance()->getConcentration() );

  //  V -= C0.getSubstance()->getConcentration();
  V -= C1.getSubstance()->getConcentration();
  
  V *= N_A * getSuperSystem()->getVolume();
  P0.getSubstance()->setQuantity( V );
''' )
