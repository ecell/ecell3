//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//       This file is part of the E-Cell LoggerBroker
//
//       Copyright (C) 1996-2009 Keio University
//       Copyright (C) 2005-2008 The Molecular Sciences Institute
//
//::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
//
//
// E-Cell LoggerBroker is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
// 
// E-Cell LoggerBroker is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public
// License along with E-Cell LoggerBroker -- see the file COPYING.
// If not, write to the Free Software Foundation, Inc.,
// 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
// 
//END_HEADER
//
// written by Koichi Takahashi
// modified by Moriyoshi Koizumi
//

#define BOOST_TEST_MODULE "LoggerBroker"

#include <boost/mpl/list.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/test/unit_test_suite.hpp>
#include <boost/test/test_case_template.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "Model.hpp"
#include "LoggerBroker.hpp"
#include "Variable.hpp"
#include "System.hpp"
#include "Exceptions.hpp"
#include "dmtool/ModuleMaker.hpp"
#include "dmtool/DMObject.hpp"

#include <algorithm>

BOOST_AUTO_TEST_CASE(testNonExistent)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;
    Model model( mmaker );
    model.setup();

    BOOST_CHECK_THROW(
        model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Value" ),
            Logger::Policy() ),
        NotFound );
}

BOOST_AUTO_TEST_CASE(testIteration)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;

    Model model( mmaker );
    model.setup();

    model.createEntity( "Variable", FullID( "Variable:/:test" ) );
    Entity* var( model.getEntity( FullID( "Variable:/:test" ) ) );
    Logger* valueLogger( model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Value" ),
            Logger::Policy() ) );
    LoggerBroker::LoggersPerFullID loggers( var->getLoggers() );
    std::vector< Logger* > logVec;
    std::copy( loggers.begin(), loggers.end(), std::back_inserter( logVec ) );
    BOOST_CHECK_EQUAL( 1, logVec.size() );
    BOOST_CHECK_EQUAL( valueLogger, logVec[ 0 ] );
    Logger* velocityLogger( model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Velocity" ),
            Logger::Policy() ) );
    logVec.clear();
    std::copy( loggers.begin(), loggers.end(), std::back_inserter( logVec ) );
    BOOST_CHECK_EQUAL( 2, logVec.size() );
    BOOST_CHECK( logVec.end() != std::find( logVec.begin(), logVec.end(), valueLogger ) );
    BOOST_CHECK( logVec.end() != std::find( logVec.begin(), logVec.end(), velocityLogger ) );
    valueLogger->log( 1.0 );
    valueLogger->log( 2.0 );
    valueLogger->log( 3.0 );
    DataPointVectorSharedPtr vec( valueLogger->getData() );
    BOOST_CHECK( DataPoint( 0.0, 0.0 ) == vec->asShort( 0 ) );
    BOOST_CHECK( DataPoint( 1.0, 0.0 ) == vec->asShort( 1 ) );
    BOOST_CHECK( DataPoint( 2.0, 0.0 ) == vec->asShort( 2 ) );
    BOOST_CHECK( DataPoint( 3.0, 0.0 ) == vec->asShort( 3 ) );
}

BOOST_AUTO_TEST_CASE(testValid)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;

    Model model( mmaker );
    model.setup();

    model.createEntity( "Variable", FullID( "Variable:/:test" ) );
    Entity* var( model.getEntity( FullID( "Variable:/:test" ) ) );
    Logger* valueLogger( model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Value" ),
            Logger::Policy() ) );
    Logger* velocityLogger( model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Velocity" ),
            Logger::Policy() ) );

    {
        LoggerBroker::const_iterator i(
            std::find( model.getLoggerBroker().begin(),
                       model.getLoggerBroker().end(),
                       LoggerBroker::iterator::value_type(
                            FullPN( "Variable:/:test:Value" ),
                            valueLogger ) ) );

        BOOST_CHECK( i != model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i(
            std::find( model.getLoggerBroker().begin(),
                       model.getLoggerBroker().end(),
                       LoggerBroker::iterator::value_type(
                            FullPN( "Variable:/:test:Velocity" ),
                            valueLogger ) ) );

        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i(
            std::find( model.getLoggerBroker().begin(),
                       model.getLoggerBroker().end(),
                       LoggerBroker::iterator::value_type(
                            FullPN( "Variable:/:test:Value" ),
                            velocityLogger ) ) );

        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i(
            std::find( model.getLoggerBroker().begin(),
                       model.getLoggerBroker().end(),
                       LoggerBroker::iterator::value_type(
                            FullPN( "Variable:/:test:Velocity" ),
                            velocityLogger ) ) );

        BOOST_CHECK( i != model.getLoggerBroker().end() );
    }
}

BOOST_AUTO_TEST_CASE(testPastTheEnd)
{
    using namespace libecs;

    ModuleMaker< EcsObject > mmaker;

    Model model( mmaker );
    model.setup();

    BOOST_CHECK( model.getLoggerBroker().begin() ==
                 model.getLoggerBroker().end() );

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().begin() );
        ++i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().begin() );
        --i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().end() );
        --i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    model.createEntity( "Variable", FullID( "Variable:/:test" ) );
    Entity* var( model.getEntity( FullID( "Variable:/:test" ) ) );
    Logger* valueLogger( model.getLoggerBroker().createLogger(
            FullPN( "Variable:/:test:Value" ),
            Logger::Policy() ) );

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
        --i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().begin() );
        --i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
        --i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
        --i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
    }

    {
        LoggerBroker::const_iterator i( model.getLoggerBroker().end() );
        --i;
        BOOST_CHECK( i == model.getLoggerBroker().begin() );
        BOOST_CHECK( i != model.getLoggerBroker().end() );
        ++i;
        BOOST_CHECK( i != model.getLoggerBroker().begin() );
        BOOST_CHECK( i == model.getLoggerBroker().end() );
    }
    
}
