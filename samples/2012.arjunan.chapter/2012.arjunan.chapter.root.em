# Wildtype MinDE model as published in
# Arjunan, S. N. V. and Tomita, M. (2010). A new multicompartmental
# reaction-diffusion modeling method links transient membrane attachment of
# E. coli MinE to E-ring formation. Syst. Synth. Biol. 4(1):35-53.
# written by Satya Arjunan <satya.arjunan(a)gmail.com>

Stepper SpatiocyteStepper(SS) { VoxelRadius 3e-8; }   # m

System System(/)
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 0;         # { 0: Cuboid (uses LENGTHX, LENGTHY, LENGTHZ)
                       #   1: Ellipsoid (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   2: Cylinder (uses LENGTHX, LENGTHY=2*radius)
                       #   3: Rod (uses LENGTHX, LENGTHY=2*radius)
                       #   4: Pyramid (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   5: Erythrocyte (uses LENGTHX, LENGTHY, LENGTHZ) }
    } 
  Variable Variable(LENGTHX)
    {
      Value 1e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 2.1e-6;      # m
    } 
  Variable Variable(LENGTHZ)
    {
      Value 2.1e-6;      # m
    } 
  Variable Variable(XZPLANE)
    {
      Value 5;      # m
    } 
  Variable Variable(XYPLANE)
    {
      Value 5;      # m
    } 
  Variable Variable(YZPLANE)
    {
      Value 3;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
    } 
  Process VisualizationLogProcess(visualize)
    {
      VariableReferenceList [_ Variable:/Sphere/Surface:VACANT]
                            [_ Variable:/Surface:VACANT]
                            [_ Variable:/Sphere:A];
      LogInterval 0.5; # s
    }
  Process MoleculePopulateProcess(populate)
    {
      VariableReferenceList [_ Variable:/Sphere:A ];
    }
}

System System(/Surface)
{
  StepperID SS;

  Variable Variable(DIMENSION)
    {
      Value 2;         # { 3: Volume
                       #   2: Surface }
    } 
  Variable Variable(VACANT)
    {
      Value 0;
    } 
}


System System(/Sphere)
{
  StepperID       SS; 
  Variable Variable(GEOMETRY)
    {
      Value 1;         # { 0: Cuboid (uses LENGTHX, LENGTHY, LENGTHZ)
                       #   1: Ellipsoid (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   2: Cylinder (uses LENGTHX, LENGTHY=2*radius)
                       #   3: Rod (uses LENGTHX, LENGTHY=2*radius)
                       #   4: Pyramid (uses LENGTHX, LENGTHY, LENGTHZ) }
                       #   5: Erythrocyte (uses LENGTHX, LENGTHY, LENGTHZ) }
    } 
  Variable Variable(LENGTHX)
    {
      Value 0.8e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 1.5e-6;      # m
    } 
  Variable Variable(LENGTHZ)
    {
      Value 1.5e-6;      # m
    } 
  Variable Variable(ORIGINX)
    {
      Value -0.3;      # m
    } 
  Variable Variable(VACANT)
    {
      Value -1; 
    } 
  Variable Variable(A)
    {
      Value 200; 
    } 
  Process DiffusionProcess(diffuse)
    {
      VariableReferenceList [_ Variable:/Sphere:A];
      D 0.1e-12;        # m^2/s
    }
}

System System(/Sphere/Surface)
{
  StepperID SS;

  Variable Variable(DIMENSION)
    {
      Value 2;         # { 3: Volume
                       #   2: Surface }
    } 
  Variable Variable(VACANT)
    {
      Value 1;
    } 
}


