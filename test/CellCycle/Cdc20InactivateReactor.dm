
CLASSNAME = 'Cdc20InactivateReactor'
BASECLASS = 'FluxReactor'
PROPERTIES = [('Real','k1',0.0), ('Real','k2',0.0), ('Real','kd',0.0)]

PROTECTED_AUX = '''
  Reactant S0;
  Reactant C0;
  Reactant C1;
'''

defineMethod( 'initialize', '''
  S0 = getReactant( "S0" );
  C0 = getReactant( "C0" );
  C1 = getReactant( "C1" );
''' )

defineMethod( 'react', '''
  const Real S( S0.getSubstance()->getConcentration() );
  const Real E2( C1.getSubstance()->getConcentration() );
  static Real v( 0 ); // static
  static Int a( 0 ); // static
  static Real time( 0 ); // static

  if(E2 >= 1)
  {
    a = 1;
  }

  if(a == 1)
  {
    time += getSuperSystem()->getStepper()->getStepInterval();
    if(time/4 <= 12)
    {
      v = k2;
    }
    if(time/4 > 12 && time/4 < 95.3)
    {
      v = k1;
    }
    if(time/4 >= 95.3)
    {
      a = 0;
      time = 0;
    }
  }

  if(a==0)
  {
    v = k2;
  }

  Real V = (v + kd) * S;
  V *= getSuperSystem()->getVolume() * N_A;

  process (V);
''' )

