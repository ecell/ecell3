CLASSNAME = 'FP25Process'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','vd',0.0),('Real','Kd',0.0)]

PROTECTED_AUX = '''
  VariableReference C0;
'''

defineMethod( 'initialize', '''
  C0 = getVariableReference( "C0" );
''' )

defineMethod( 'process', '''
  Real E( C0.getVariable()->getConcentration() );

  Real V( -1 * vd * E );
  V /= Kd + E;
  V *= 1E-018 * N_A;

  setFlux( V );
''' )
