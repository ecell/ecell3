CLASSNAME = 'Bck2PReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','Bck2',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant P0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  P0 = getReactant( "P0" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getSubstance()->getConcentration() ); // const?
  Real V( Bck2 * E1);
  V *= N_A * getSuperSystem()->getVolume();
  P0.getSubstance()->setQuantity( V );
''' )
