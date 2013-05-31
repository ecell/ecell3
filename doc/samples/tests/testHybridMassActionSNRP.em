# A test model for Spatiocyte 
# written by Satya Arjunan <satya.arjunan(a)gmail.com>

Stepper SpatiocyteStepper(SS)
{
  VoxelRadius 4.4e-9;
}

Stepper FixedODE1Stepper(DE)
{
}

System System( / )
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 3;     
    } 
  Variable Variable(LENGTHX)
    {
      Value 4.5e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 1e-6;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
    } 

  # Mass Action reference
  Variable Variable(Aa)
    {
      Value 1000; 
      Name "HD";
    } 
  Variable Variable(Ba)
    {
      Value 0;   
      Name "HD";
    } 
  Variable Variable(Ca)
    {
      Value 0;  
      Name "HD";
    } 
  # You can also replace this with MassActionFluxProcess
  Process MassActionProcess( AtoBa )
    {
      StepperID DE;
      VariableReferenceList [_ Variable:/:Aa -1]
                            [_ Variable:/:Ba 1];
      k 1e-3;
    }  
  # You can also replace this with MassActionFluxProcess
  Process MassActionProcess( BtoCa )
    {
      StepperID DE;
      VariableReferenceList [_ Variable:/:Ba -1]
                            [_ Variable:/:Ca 1];
      k 1;
    }

  # Full SpatiocyteNextReactionProcess
  Variable Variable(Ab)
    {
      Value 1000; 
      Name "HD";
    } 
  Variable Variable(Bb)
    {
      Value 0;  
      Name "HD";
    } 
  Variable Variable(Cb)
    {
      Value 0;   
      Name "HD";
    } 
  Process SpatiocyteNextReactionProcess( AtoBb )
    {
      VariableReferenceList [_ Variable:/:Ab -1]
                            [_ Variable:/:Bb 1];
      k 1e-3;
    }  
  Process SpatiocyteNextReactionProcess( BtoCb )
    {
      VariableReferenceList [_ Variable:/:Bb -1]
                            [_ Variable:/:Cb 1];
      k 1;
    }  

  # Hybrid Mass Action and SpatiocyteNextReactionProcess
  Variable Variable(Ac)
    {
      Value 1000; 
      Name "HD";
    } 
  Variable Variable(Bc)
    {
      Value 0;
      Name "HD";
    } 
  Variable Variable(Cc)
    {
      Value 0;
      Name "HD";
    } 
  Process MassActionProcess( AtoBc )
    {
      StepperID DE;
      VariableReferenceList [_ Variable:/:Ac -1]
                            [_ Variable:/:Bc 1];
      k 1e-3;
    }  
  Process SpatiocyteNextReactionProcess( BtoCc )
    {
      VariableReferenceList [_ Variable:/:Bc -1]
                            [_ Variable:/:Cc 1];
      k 1;
    }  

  # Hybrid SpatiocyteNextReactionProcess and Mass Action
  Variable Variable(Ad)
    {
      Value 1000;
      Name "HD";
    } 
  Variable Variable(Bd)
    {
      Value 0;  
      Name "HD";
    } 
  Variable Variable(Cd)
    {
      Value 0; 
      Name "HD";
    } 
  Process SpatiocyteNextReactionProcess( AtoBd )
    {
      VariableReferenceList [_ Variable:/:Ad -1]
                            [_ Variable:/:Bd 1];
      k 1e-3;
    }  
  Process MassActionProcess( BtoCd )
    {
      StepperID DE;
      VariableReferenceList [_ Variable:/:Bd -1]
                            [_ Variable:/:Cd 1];
      k 1;
    }  
}

