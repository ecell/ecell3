
CLASSNAME = 'Hct1InactivateProcess'
BASECLASS = 'FluxProcess'
PROPERTIES = [('Real','k1',0.0),('Real','k2',0.0),('Real','e1',0.0),('Real','e2',0.0),('Real','e3',0.0),('Real','J',0.0)]

PROTECTED_AUX = '''
  VariableReference S0;
  VariableReference C0;
  VariableReference C1;
  VariableReference C2;
  VariableReference C3;

  ProcessPtr theActivateProcess;

'''

defineMethod( 'initialize', '''
  S0 = getVariableReference( "S0" );
  C0 = getVariableReference( "C0" );
  C1 = getVariableReference( "C1" );
  C2 = getVariableReference( "C2" );
  C3 = getVariableReference( "C3" );

  theActivateProcess = getSuperSystem()->getProcess( "Hct1act" );

''' )

defineMethod( 'process', '''
  const Real S( S0.getVariable()->getConcentration() );
  const Real E1( C0.getVariable()->getConcentration() );
  const Real E2( C1.getVariable()->getConcentration() );
  const Real E3( C2.getVariable()->getConcentration() );
  const Real E4( C3.getVariable()->getConcentration() );
  Real v( E1 + e1 * E2 + e2 * E3 + e3 * E4 );
  v *= k2;
  v += k1;
  Real V( v * S ); 
  V /= J + S; // conc
  V += getSuperSystem()->getVolume() * N_A; // conc -> quantity


  // limit so that S0 is always >= 0

//  std::cerr << "A: " << ActivateActivity;
//  std::cerr << "V: " << V;
  Real S0ValuePerSec( S0.getVariable()->getValue() 
                      / getStepper()->getStepInterval() );


  const Real ActivateActivity( theActivateProcess->getActivity() );

  if( S0ValuePerSec - V + ActivateActivity < 0 ) 
//  if( S0ValuePerSec - V < 0 )
  {
    // calculate the point where S0 = 0
    V = S0ValuePerSec + ActivateActivity - 10E-6;
  //  std::cerr << "limited V: " << V << std::endl;
  } 



  setFlux( V );
''' )

