
CLASSNAME = 'SPNReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getSubstance()->getConcentration() );
  Real V( k * E1 );
  V /= J + E1;
  V *= getSuperSystem()->getVolume() * N_A;

  process ( V );
''' )

