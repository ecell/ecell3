CLASSNAME = 'Cln3PReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','Max',0.0),('Real','D',0.0),('Real','J',0.0)]

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
 Real V( Max * D * E1 );
 V /= J + D * E1;
 V *= getSuperSystem()->getVolume() * N_A;
 P0.getSubstance()->setQuantity(V);
''' )
