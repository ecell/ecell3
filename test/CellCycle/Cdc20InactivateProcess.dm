
CLASSNAME = 'Cdc20InactivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0), ('Real','k2',0.0), ('Real','kd',0.0)]

PROTECTED_AUX = '''
  Connection S0;
  Connection C0;
  Connection C1;
  Real v; // declare member variable
  Int a; // declare member variable
  Real time; // declare member variable
'''

defineMethod( 'initialize', '''
  S0 = getConnection( "S0" );
  C0 = getConnection( "C0" );
  C1 = getConnection( "C1" );
  v = 0; // initialize member variable
  a = 0; // initialize member variable
  time = 0; // initialize member variable
''' )

defineMethod( 'process', '''
  const Real S( S0.getVariable()->getConcentration() );
  const Real E2( C1.getVariable()->getConcentration() );
  // Real v( 0 ); // removed static 
  // Int a( 0 ); // removed static 
  // Real time( 0 ); // removed static 

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

  setFlux(V);
''' )

