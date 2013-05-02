//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//                This file is part of E-Cell Simulation Environment package
//
//                                Copyright (C) 2006-2009 Keio University
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Satya Arjunan <satya.arjunan@gmail.com>
// E-Cell Project, Institute for Advanced Biosciences, Keio University.
//

#include <H5Cpp.h>
#include <H5Support.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>
#include <boost/mpl/and.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_signed.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <boost/type_traits/make_unsigned.hpp>
#include <SpatiocyteProcess.hpp>
#include <SpatiocyteSpecies.hpp>

class ParticleData
{
public:
    unsigned int getCoordinate() const
    {
        return theCoord;
    }

    unsigned short getSpeciesID() const
    {
        return theSpeciesID;
    }

    unsigned int getLatticeID() const
    {
        return theLatticeID;
    }

    ParticleData(unsigned int coord, unsigned short speciesID, unsigned int latticeID)
        : theCoord(coord), theSpeciesID(speciesID), theLatticeID(latticeID) {}

private:
    const unsigned int theCoord;
    const unsigned short theSpeciesID;
    const uint64_t theLatticeID;
};


template<typename Tarc>
struct PointDataPacker
{
    typedef Tarc archiver_type;

    void operator()(archiver_type& arc, Point const* data = 0) const
    {
        arc << field("x", &Point::x, data);
        arc << field("y", &Point::y, data);
        arc << field("z", &Point::z, data);
    }
};

template<typename Tarc>
struct ParticleDataPacker
{
    typedef Tarc archiver_type;

    void operator()(archiver_type& arc, ParticleData const* data = 0) const
    {
        arc << field<uint64_t>("id", &ParticleData::getCoordinate, data);
        arc << field<uint64_t>("species_id", &ParticleData::getSpeciesID, data);
        arc << field<uint64_t>("lattice_id", &ParticleData::getLatticeID, data);
    }
};

/* XXX: not reentrant, but it should be ok */
template<typename Tarc>
struct SpeciesPacker
{
    typedef Tarc archiver_type;

    static std::string Species_getName(Species const& self)
    {
        std::string name;
        libecs::Variable const* variable(self.getVariable());
        name = variable->getName();
        if (name.empty())
        {
            name = variable->getID();
        }
        return name;
    }

    void operator()(archiver_type& arc, Species const* data = 0) const
    {
        arc << field<uint64_t>("id", &Species::getID, data);
        arc << field<char[32]>("name", &Species_getName, data);
        arc << field<double>("radius", &Species::getMoleculeRadius, data);
        arc << field<double>("D", &Species::getDiffusionCoefficient, data);
    }
};

template<typename Tarc>
struct CompPacker
{
    typedef Tarc archiver_type;

    static boost::array<double, 3> Comp_lengths(Comp const& Comp)
    {
        boost::array<double, 3> retval;
        retval[0] = Comp.lengthX;
        retval[1] = Comp.lengthY;
        retval[2] = Comp.lengthZ;
        return retval;
    }

    void operator()(archiver_type& arc, Comp const* data = 0) const
    {
        arc << field<uint8_t>("id", &Comp::vacantID, data);
        arc << field<boost::array<double, 3> >("lengths", &Comp_lengths, data);
        arc << field<Comp, double, double>("voxelRadius", voxelRadius);
        arc << field<Comp, double, double>("normalizedVoxelRadius", normalizedVoxelRadius);
        arc << field<Comp, unsigned int, unsigned int>("startCoord", startCoord);
        arc << field<Comp, unsigned int, unsigned int>("layerSize", layerSize);
        arc << field<Comp, unsigned int, unsigned int>("rowSize", rowSize);
        arc << field<Comp, unsigned int, unsigned int>("colSize", colSize);
    }

    CompPacker(double voxelRadius, double normalizedVoxelRadius,
               unsigned int startCoord, unsigned int layerSize,
               unsigned int rowSize, unsigned int colSize)
        : voxelRadius(voxelRadius), normalizedVoxelRadius(normalizedVoxelRadius),
          startCoord(startCoord), layerSize(layerSize), rowSize(rowSize), colSize(colSize) {}

    CompPacker(): voxelRadius(0.), normalizedVoxelRadius(0.),
                  startCoord(0), layerSize(0), rowSize(0), colSize(0) {}

private:
    const double voxelRadius;
    const double normalizedVoxelRadius;
    unsigned int startCoord;
    unsigned int layerSize;
    unsigned int rowSize;
    unsigned int colSize;
};

template<template<typename> class TTserialize_>
H5::CompType getH5Type()
{
    return get_h5_type<h5_le_traits, TTserialize_>();
}

template<typename T>
unsigned char* pack(unsigned char* buffer, T const& container)
{
    return pack<h5_le_traits>(buffer, container);
}

template<template<typename> class TTserialize_, typename T>
unsigned char* pack(unsigned char* buffer, T const& container)
{
    return pack<h5_le_traits, TTserialize_>(buffer, container);
}

LIBECS_DM_CLASS(H5VisualizationLogProcess, SpatiocyteProcess)
{ 
public:
    typedef h5_le_traits traits_type;
    typedef traits_type::packer_type packer_type;

public:
    LIBECS_DM_OBJECT(H5VisualizationLogProcess, Process)
    {
        INHERIT_PROPERTIES(Process);
        PROPERTYSLOT_SET_GET(Integer, Polymer);
        PROPERTYSLOT_SET_GET(Real, LogInterval);
        PROPERTYSLOT_SET_GET(String, FileName);
    }

    H5VisualizationLogProcess();

    virtual ~H5VisualizationLogProcess() {}
    SIMPLE_SET_GET_METHOD(Integer, Polymer);
    SIMPLE_SET_GET_METHOD(Real, LogInterval);
    SIMPLE_SET_GET_METHOD(String, FileName);

    virtual void initialize()
      {
        if(isInitialized)
          {
            return;
          }
        SpatiocyteProcess::initialize();
        isPriorityQueued = true;
    }
    virtual void initializeFifth()
    {
        if(LogInterval > 0)
        {
            theInterval = LogInterval;
        }
        else
        {
            //Use the smallest time step of all queued events for
            //the step interval:
            theTime = libecs::INF;
            thePriorityQueue->move(theQueueID);
            theInterval = thePriorityQueue->getTop()->getTime();
        }
        theTime = theInterval;
        thePriorityQueue->move(theQueueID);
    }

    virtual void initializeLastOnce()
    {
        theLogFile = H5::H5File(FileName, H5F_ACC_TRUNC);
        theDataGroup = theLogFile.createGroup("data");
        initializeLog();
        logSpecies();
    }

    virtual void fire()
    {
        logSpecies();
        if(LogInterval > 0)
        {
            theTime += LogInterval;
            thePriorityQueue->moveTop();
        }
        else
        {
            //get the next step interval of the SpatiocyteStepper:
            double aTime(theTime);
            theTime = libecs::INF;
            thePriorityQueue->moveTop();
            if(thePriorityQueue->getTop()->getTime() > aTime)
            {
                theInterval = thePriorityQueue->getTop()->getTime() -
                    theSpatiocyteStepper->getCurrentTime();
            }
            theTime = aTime + theInterval;
            thePriorityQueue->move(theQueueID);
        }
    }
protected:
    void initializeLog();
    void logSpecies();
    void logMolecules(H5::DataSpace const& space, H5::DataSet const& dataSet, hsize_t (&dims)[1], Species *);

    template<typename T>
    void setH5Attribute(H5::Group& dg, const char* name, T const& data);
    void setH5Attribute(H5::Group& dg, const char* name, Point const& data);

protected:
    unsigned int Polymer;
    unsigned int theLogMarker;
    double LogInterval;
    String FileName;
    packer_type packer;
    H5::H5File theLogFile;
    H5::Group theDataGroup;
    H5::CompType pointDataType;
    H5::CompType particleDataType;
    H5::CompType speciesDataType;
    H5::CompType CompDataType;
};

H5VisualizationLogProcess::H5VisualizationLogProcess()
    :   SpatiocyteProcess(),
        Polymer(1),
        theLogMarker(UINT_MAX),
        LogInterval(0),
        FileName("visualLog.h5"),
        pointDataType(getH5Type<PointDataPacker>()),
        particleDataType(getH5Type<ParticleDataPacker>()),
        speciesDataType(getH5Type<SpeciesPacker>()),
        CompDataType(getH5Type<CompPacker>())
{
}

template<typename T>
void H5VisualizationLogProcess::setH5Attribute(H5::Group& dg, const char* name, T const& data)
{
    H5::DataType dataType = get_h5_scalar_data_type_le<T>()();
    unsigned char buf[sizeof(T)];
    packer(buf, data);
    dg.createAttribute(name, dataType, H5::DataSpace()).write(dataType, buf); 
}

void H5VisualizationLogProcess::setH5Attribute(H5::Group& dg, const char* name, Point const& data)
{
    H5::Attribute attr(dg.createAttribute(name, pointDataType, H5::DataSpace()));
    unsigned char buf[24];
    BOOST_ASSERT(sizeof(buf) >= pointDataType.getSize());
    pack<PointDataPacker>(buf, data);
    attr.write(pointDataType, buf); 
}


void H5VisualizationLogProcess::initializeLog()
{
    {
        std::vector<Comp*> const& Comps(theSpatiocyteStepper->getComps());
        const hsize_t dims[] = { Comps.size() };
        boost::scoped_array<unsigned char> buf(new unsigned char[Comps.size() * CompDataType.getSize()]);
        field_packer<h5_le_traits> packer(buf.get());
        CompPacker<field_packer<h5_le_traits> > serializer(
            theSpatiocyteStepper->getVoxelRadius(),
            theSpatiocyteStepper->getNormalizedVoxelRadius(),
            theSpatiocyteStepper->getStartCoord(),
            theSpatiocyteStepper->getLayerSize(),
            theSpatiocyteStepper->getRowSize(),
            theSpatiocyteStepper->getColSize());
        H5::Group latticeInfoGroup(theLogFile.createGroup("lattice_info"));
        H5::DataSpace space(H5::DataSpace(1, dims, dims));
        H5::DataSet latticeInfoDataSet(latticeInfoGroup.createDataSet("HCP_group", CompDataType, space));
        BOOST_FOREACH(Comp const* Comp, Comps)
        {
            serializer(packer, Comp);
        }
        latticeInfoDataSet.write(buf.get(), CompDataType, space);
    }

    {
        const hsize_t dims[] = { theProcessSpecies.size() };
        boost::scoped_array<unsigned char> buf(new unsigned char[speciesDataType.getSize() * theProcessSpecies.size()]);
        H5::DataSpace space(H5::DataSpace(1, dims, dims));
        H5::DataSet speciesSet(theLogFile.createDataSet("species", speciesDataType, space));
        unsigned char* p(buf.get());
        BOOST_FOREACH(Species const* species, theProcessSpecies)
        {
            p = pack<SpeciesPacker>(p, *species);
        }
        speciesSet.write(buf.get(), speciesDataType, space);
    }
}

void H5VisualizationLogProcess::logMolecules(H5::DataSpace const& space, H5::DataSet const& dataSet, hsize_t (&dims)[1], Species* aSpecies)
{
    //No need to log lipid or vacant molecules since we have
    //already logged them once during initialization:
    if(aSpecies->getIsVacant())
    {
      if(aSpecies->getIsDiffusiveVacant() || aSpecies->getIsReactiveVacant())
        {
          aSpecies->updateMolecules();
        }
      else
        {
          return;
        }
    }
    //theLogFile.write((char*)(&anIndex), sizeof(anIndex));
    //The species molecule size:
    const int aSize(aSpecies->size());

    if (!aSize)
        return;

    hsize_t wdim[] = { aSize };
    const hsize_t offset[] = { dims[0] };

    dims[0] += wdim[0];
    dataSet.extend(dims);

    H5::DataSpace mem(1, wdim);
    H5::DataSpace slab(dataSet.getSpace());
    slab.selectHyperslab(H5S_SELECT_SET, wdim, offset);
    boost::scoped_array<unsigned char> buf(new unsigned char[particleDataType.getSize() * aSize]);
    unsigned char* p = buf.get();
    for(int i(0); i != aSize; ++i)
    {
        unsigned int const coord(aSpecies->getCoord(i));
        p = pack<ParticleDataPacker>(p, ParticleData(coord, aSpecies->getID(), aSpecies->getComp()->vacantSpecies->getID()));
    }
    dataSet.write(buf.get(), particleDataType, mem, slab);
}

void H5VisualizationLogProcess::logSpecies()
{
    const double currentTime(theSpatiocyteStepper->getCurrentTime());

    H5::Group perTimeDataGroup(theDataGroup.createGroup(boost::lexical_cast<std::string>(currentTime).c_str()));
    setH5Attribute(perTimeDataGroup, "t", currentTime);
    H5::DataSpace space;
    H5::DataSet dataSet;
    {
        static const hsize_t initdims[] = { 0 };
        static const hsize_t maxdims[] = { H5S_UNLIMITED };
        static const hsize_t chunkdims[] = { 128 };
        space = H5::DataSpace(1, initdims, maxdims);
        H5::DSetCreatPropList props;
        props.setChunk(1, chunkdims);
        dataSet = perTimeDataGroup.createDataSet("particles", particleDataType, space, props);
    }

    hsize_t dims[] = { 0 };

    for(unsigned int i(0); i != theProcessSpecies.size(); ++i)
    {
        logMolecules(space, dataSet, dims, theProcessSpecies[i]);
    }
}

LIBECS_DM_INIT(H5VisualizationLogProcess, Process); 
