
CLASSNAME = 'CyclinsynthesisReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
  Reactant C1;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
''' )

defineMethod( 'react', '''
  const Real E1( C0.getSubstance()->getConcentration() ); 
  const Real E2( C1.getSubstance()->getConcentration() ); 
  Real V( k1 + k2 * E1 );

  V *= E2;
  V *= getSuperSystem()->getVolume() * N_A;

  process( V );
''' )

