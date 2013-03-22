#include <libecs/libecs.hpp>

#include <fstream>
#include <libecs/MethodProxy.hpp>

#include "IteratingLogProcess.hpp"
#include "SpatiocyteSpecies.hpp"

USE_LIBECS;

LIBECS_DM_CLASS( IntensityLoggerProcess, IteratingLogProcess )
{ 
public:

    LIBECS_DM_OBJECT( IntensityLoggerProcess, Process )
    {
        INHERIT_PROPERTIES( IteratingLogProcess );

        PROPERTYSLOT_SET_GET( Real, AxialRadius );
        PROPERTYSLOT_SET_GET( Real, RadialRadius );
        PROPERTYSLOT_SET_GET( Real, PeakIntensity );
    }

    IntensityLoggerProcess()
        :
        r_axial( 500e-9 ),
        r_radial( 200e-9 ),
        I0( 1.0 )
    {
        ; // do nothing
    }

    GET_METHOD( Real, AxialRadius ) { return r_axial; }
    GET_METHOD( Real, RadialRadius ) { return r_radial; }
    GET_METHOD( Real, PeakIntensity ) { return I0; }

    SET_METHOD( Real, AxialRadius ) { r_axial = value; }
    SET_METHOD( Real, RadialRadius ) { r_radial = value; }
    SET_METHOD( Real, PeakIntensity ) { I0 = value; }

    virtual ~IntensityLoggerProcess()
    {
        ; // do nothing
    }

    virtual void initializeLastOnce()
    {
        theLogFile.open( FileName.c_str(), ios::trunc );
        theLogFile.setf( ios::scientific );
        theLogFile.precision( 16 );
        initializeLog();
        logSpecies();
    }

    virtual void fire()
    {
        if ( theTime <= LogDuration )
        {
            logSpecies();
            theTime += theStepInterval;
        }
        else
        {
            theTime = libecs::INF;
            theLogFile.flush();
            theLogFile.close();
        }

        thePriorityQueue->moveTop();
    }

    void logSpecies()
    {
        const double r_radial_sq( r_radial * r_radial );
        const double r_axial_sq( r_axial * r_axial );

        const double voxel_size( theSpatiocyteStepper->getVoxelRadius() * 2 );
        const double voxel_size_sq( voxel_size * voxel_size );

        const Point center_point( theSpatiocyteStepper->getCenterPoint() );
        double intensity( 0.0 );

        for ( unsigned int i( 0 ); i != theProcessSpecies.size(); ++i )
        {
            Species* aSpecies( theProcessSpecies[ i ] );
            for ( unsigned int j( 0 ); j != aSpecies->size(); ++j )
            {
                unsigned int coord( aSpecies->getCoord( j ) );
                Point aPoint( theSpatiocyteStepper->coord2point( coord ) );

                const double x( aPoint.x - center_point.x );
                const double y( aPoint.y - center_point.y );
                const double r_sq( fabs( x * x + y * y ) * voxel_size_sq );
                const double z( fabs( aPoint.z - center_point.z ) );
                const double z_sq( z * z * voxel_size_sq );

                intensity += I0 * exp( -2 * r_sq / r_radial_sq ) 
                    * exp( -2 * z_sq / r_axial_sq );
            }
        }

        theLogFile << theSpatiocyteStepper->getCurrentTime() 
                   << " " << intensity << endl;
    }

protected:

    void initializeLog()
    {
        ; // do nothing

//         Point aCenterPoint( theSpatiocyteStepper->getCenterPoint() );
//         theLogFile
//             << "startCoord:" << theSpatiocyteStepper->getStartCoord()
//             << " rowSize:" << theSpatiocyteStepper->getRowSize() 
//             << " layerSize:" << theSpatiocyteStepper->getLayerSize()
//             << " colSize:" << theSpatiocyteStepper->getColSize()
//             << " width:" << aCenterPoint.z * 2
//             << " height:" << aCenterPoint.y * 2
//             << " length:" <<  aCenterPoint.x * 2
//             << " voxelRadius:" << theSpatiocyteStepper->getVoxelRadius()
//             << " moleculeSize:" << theMoleculeSize << endl;
    }

protected:

    Real r_axial, r_radial, I0;

};

LIBECS_DM_INIT( IntensityLoggerProcess, Process );
