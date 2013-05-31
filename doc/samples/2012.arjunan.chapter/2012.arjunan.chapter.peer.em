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
      Value 3e-6;      # m
    } 
  Variable Variable(LENGTHY)
    {
      Value 3e-6;      # m
    } 
  Variable Variable(LENGTHZ)
    {
      Value 3e-6;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 0; 
    } 
  Process VisualizationLogProcess(visualize)
    {
      VariableReferenceList [_ Variable:/Sphere/Surface:VACANT]
                            [_ Variable:/Rod/Surface:VACANT]
                            [_ Variable:/Rod:B]
                            [_ Variable:/Sphere:A];
      LogInterval 0.5; # s
    }
  Process MoleculePopulateProcess(populate)
    {
      VariableReferenceList [_ Variable:/Sphere:A ]
                            [_ Variable:/Rod:B ];
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
      Value 1.5e-6;      # m
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
      Value -0.32;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 4; 
    } 
  Variable Variable(A)
    {
      Value 400; 
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
      Value 0;
    } 
}


System System(/Rod)
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
      Value 1.5e-6;      # m
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
      Value 0.31;      # m
    } 
  Variable Variable(VACANT)
    {
      Value 5; 
    } 
  Variable Variable(B)
    {
      Value 400; 
    } 
  Process DiffusionProcess(diffuse)
    {
      VariableReferenceList [_ Variable:/Rod:B];
      D 0.1e-12;        # m^2/s
    }
}

System System(/Rod/Surface)
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
