CLASSNAME = 'FP25Reactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','vd',0.0),('Real','Kd',0.0)]

PROTECTED_AUX = '''
  Reactant C0;
'''

defineMethod( 'initialize', '''
  C0 = getReactant( "C0" );
''' )

defineMethod( 'react', '''
  Real E( C0.getSubstance()->getConcentration() );

  Real V( -1 * vd * E );
  V /= Kd + E;
  V *= 1E-018 * N_A;

  process( V );
''' )
