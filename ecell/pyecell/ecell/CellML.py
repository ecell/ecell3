# -*- coding: utf-8 -*-
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#       This file is part of the E-Cell System
#
#       Copyright (C) 1996-2014 Keio University
#       Copyright (C) 2008-2014 RIKEN
#       Copyright (C) 2005-2009 The Molecular Sciences Institute
#
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#
#
# E-Cell System is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either
# version 2 of the License, or (at your option) any later version.
# 
# E-Cell System is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public
# License along with E-Cell System -- see the file COPYING.
# If not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
# 
#END_HEADER

__program__   = 'CellML'
__version__   = '0.1'
__author__    = 'Yasuhiro Naito <ynaito@e-cell.org>'
__copyright__ = 'Keio University, RIKEN'
__license__   = 'GPL'

## ------------------------------------------
## 定数：名前空間
## ------------------------------------------
CELLML_NAMESPACE_1_0  = 'http://www.cellml.org/cellml/1.0#'
CELLML_NAMESPACE_1_1  = 'http://www.cellml.org/cellml/1.1#'
MATHML_NAMESPACE      = 'http://www.w3.org/1998/Math/MathML'


## ------------------------------------------
## 定数：方程式の型
## ------------------------------------------
CELLML_MATH_ALGEBRAIC_EQUATION  = 0
CELLML_MATH_ASSIGNMENT_EQUATION = 1
CELLML_MATH_RATE_EQUATION       = 2

CELLML_MATH_LEFT_SIDE  = 10
CELLML_MATH_RIGHT_SIDE = 11

CELLML_MATH_NODE = 20


import xml.etree.ElementTree as et
from xml.etree.ElementTree import XMLParser

from math import floor, ceil, factorial, exp, log, log10, \
                 sin, cos, tan, asin, acos, atan, sqrt, pow
import numbers
from copy import deepcopy


##=====================================================================================================
##  CellMLクラス
##=====================================================================================================

class CellML( object ):

    ##----クラス内クラス--------------------------------------------------------------

    class component( object ):
        
        def __init__( self, name, variables = [], maths = [], reactions = [], 
                            units = [], meta_id = '', 
                            parent = '', children = [] ):
            
            self.name        = str( name )
            self.variables   = variables
            self.maths       = maths
            self.reactions   = reactions
            self.units       = units
            self.meta_id     = str( meta_id )
            self.parent      = str( parent )
            self.children    = children

    class global_variable( object ):
    # component間のvariableのconnection（同一関係）を解決し、
    # モデル中に「実存」するvariableを格納するクラス。
    # local_variableとの対応関係も格納する。
        
        def __init__( self, name, component, initial_value = None, 
                      connection = [], units = None, meta_id = '' ):
            self.name              = str( name )
            self.component         = str( component )
            self.initial_value     = initial_value
            self.connection        = connection
            self.units             = units
            self.meta_id           = str( meta_id )
        
        def has_initial_value( self ):
            return isinstance( self.initial_value, numbers.Number )

    class local_variable( object ):
    # component中で定義されるvariableの属性を格納するクラス
        
        def __init__( self, name, initial_value = None, 
                      public_interface = 'none', private_interface = 'none', 
                      units = None, connection = False, meta_id = '' ):
            self.name              = str( name )
            self.initial_value     = initial_value
            self.public_interface  = str( public_interface )
            self.private_interface = str( private_interface )
            self.connection        = connection
            self.units             = units
            self.meta_id           = str( meta_id )
        
        def has_initial_value( self ):
            return isinstance( self.initial_value, numbers.Number )

    class variable_address( object ):
    # componentと、その中でのvariable名のペア。
    # モデル中のユニークなひとつのvariableを指すIDとして機能する。
        
        def __init__( self, component, name ):
            
            self.component = str( component )
            self.name      = str( name )
            self.ID        = '{0.component}:{0.name}'.format( self )

    class global_math( object ):
    # component中のmath（local_math）のvariableを、対応する
    # global_variableのvaに置き換えたオブジェクトを格納するためのクラス。
    
        def __init__( self, component, math ):
            
            self.component  = str( component )             ## component名
            self.local_math = math                         ## MathMLオブジェクト（local_variable名）
            self.math       = deepcopy( self.local_math )  ## MathMLオブジェクト（global_variableに置換）
            self.tag        = self.math.tag
            self.tag_group  = self.math.tag_group
        
        def get_expression_str( self ):
            return self.math.get_expression_str()

    class divided_ode( object ):
    # mathエレメントの多項式を分解し、math間の共通項を抽出して、
    # 量論係数とともに格納するためのクラス。
    
        def __init__( self, component, math, variable = None, genuine = 0, coefficient = 1 ):
            
            self.component          = str( component )       ## component名
            self.math               = math                   ## MathMLオブジェクト
            self.variable           = variable               ## mathの対象となる変数
            self.genuine            = genuine                ## 重複関係で消去対象か？
            self.coefficient        = coefficient            ## この項全体の係数。最終的にはstoichiometry_listに落とし込んで1に戻す
            self.tag                = self.math.tag
            self.tag_group          = self.math.tag_group
            self.stoichiometry_list = self._init_stoichiometry_list()  ## 式に含まれるすべての変数（ciエレメント）を量論係数0で格納したリスト
        
        def _init_stoichiometry_list( self ):
            
            # 式に含まれるglobal_variableの重複のないリスト_ci_lsを作成する。
            if self.math.root_node.tag == self.tag[ 'ci' ]:
                _ci_ls = self.math.root_node.text.split( ':' )
            else:
                _ci_ls = list( set( [ tuple( ci.text.split( ':' ) ) for ci in self.math.root_node.findall( './/' + self.tag[ 'ci' ] ) ] ) )
            
            return [ CellML.stoichiometry( CellML.variable_address( _ci[ 0 ], _ci[ 1 ] ), 0 ) for _ci in _ci_ls ]
        
        def _set_stoichiometry( self, va, coefficient ):
            
            # va は、variable_addressオブジェクト または
            # "component:name" 形式の文字列（divided_ode.variableはこの形式）
            
            if isinstance( va, str ):
                va = va.split( ':' )
                va = CellML.variable_address( va[ 0 ], va[ 1 ] )
            
            _sc_ls =[ _sc for _sc in self.stoichiometry_list if CellML._is_same_variable_address( _sc.variable_address, va ) ]
            
            if len( _sc_ls ) != 1:
                raise TypeError, "{0} must be included just once in '{1}'".format( self.variable, self.math.do.math.get_expression_str() )
            
            _sc_ls.pop().coefficient = coefficient

    class stoichiometry( object ):
    # divided_ode.stoichiometry_listの要素
    # 反応に関わるvariableとその量論係数のペアを格納する。
    
        def __init__( self, variable_address, coefficient ):
            
            self.variable_address = variable_address  ## variable_address
            self.coefficient      = coefficient     ## 量論係数（int）


    ##----初期化----------------------------------------------------------------------

    def __init__( self, CellML_file_path ):
        
        parser = XMLParser()
        parser.feed( open( CellML_file_path, 'r' ).read() )

        self.root_node = parser.close()  ## xml.etree.ElementTree.Element object

        # エレメントのテキストの冒頭、末尾の空白を削除する。
        self.root_node.text = str( self.root_node.text ).strip()
        for _e in self.root_node.findall( './/*' ):
            _e.text = str( _e.text ).strip()

        if self.root_node.tag == '{http://www.cellml.org/cellml/1.0#}model':
            self.namespace = 'http://www.cellml.org/cellml/1.0#'
        elif self.root_node.tag == '{http://www.cellml.org/cellml/1.1#}model':
            self.namespace =  'http://www.cellml.org/cellml/1.1#'
        else:
            raise TypeError, "CellML version is not specified."

        self.tag = dict(
            
            component        = '{%s}component' % self.namespace,
            variable         = '{%s}variable' % self.namespace,
            group            = '{%s}group' % self.namespace,
            relationship_ref = '{%s}relationship_ref' % self.namespace,
            component_ref    = '{%s}component_ref' % self.namespace,
            connection       = '{%s}connection' % self.namespace,
            map_components   = '{%s}map_components' % self.namespace,
            map_variables    = '{%s}map_variables' % self.namespace,

            math       = '{{{}}}math'.format( MATHML_NAMESPACE ),
            apply      = '{{{}}}apply'.format( MATHML_NAMESPACE ),
            math_apply = '{{{0}}}math/{{{0}}}apply'.format( MATHML_NAMESPACE ),
        )

        ##----モデル構造の抽出-------------

        self.components = []
        self.variable_attributes = ( 'initial_value', 'public_interface', 'private_interface', 'units' )
        self.containment_hierarchies = {} ## encapsulation は要素間の隠蔽関係の定義なので、E-Cell 3 では記述対象外
        self.connections = []
        self.global_variables = []
        self.global_maths = []
        self.divided_odes = []

        self._get_components()
        self._dump_components()

        self._get_containment_hierarchies()
        self._dump_containment_hierarchies()

        self._get_connections()
#        self._dump_connections()

        self._get_global_variables()
#        self._dump_global_variables()

        ##----初期値の計算----------------

        self._calc_initial_values()
        self._dump_global_variables()
#
        self._dump_global_maths()

        ##----微分方程式の分割------------
        self._divide_polynomial_ode()


    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _get_components( self ):
        
        for component_node in self.root_node.iterfind( './/' + self.tag[ 'component' ] ):
            
            if not self._has_name( component_node ):
                raise TypeError, "Component must have name attribute."
            else:
                _component_name = component_node.get( 'name' )
            
            ## variables
            _variables = []
            
            for variable_node in component_node.iterfind( './' + self.tag[ 'variable' ] ):
                
                if not self._has_name( variable_node ):
                    raise TypeError, "Variable must have name attribute. ( in component: %s )" % component_node.get( 'name' )
                
                _variables.append( self.get_local_variable( variable_node ) )
            
                # self._update_variable( variable_node )
            
            ## maths
            _maths = []
            
            for eq in component_node.findall( './' + self.tag[ 'math_apply' ] ):
                _MathML = MathML( eq )
                _maths.append( _MathML )
                self.global_maths.append( self.global_math( _component_name, deepcopy( _MathML ) ) )
            
            ## reactions
            _reactions = []
            
            ###############################
            ##                           ##
            ##   reaction の処理は未実装   ##
            ##                           ##
            ###############################
            
            ## register component object
            self.components.append( self.component( 
                _component_name ,
                _variables, 
                _maths, 
                _reactions ) )

    ##-------------------------------------------------------------------------------------------------
    def get_local_variable( self, variable_node ):
        
        ## 後方からpop()していくための並び順
        _variable_attributes = ( 'units', 'private_interface', 'public_interface', 'initial_value', 'name' )
        
        _attributes = [ variable_node.get( attrib ) for attrib in _variable_attributes ]
        
        return self.local_variable( 
                   _attributes.pop(),    # name
                   _attributes.pop(),    # initial_value
                   _attributes.pop(),    # public_interface
                   _attributes.pop(),    # private_interface
                   _attributes.pop(), )  # units

    ##-------------------------------------------------------------------------------------------------
    def _has_name( self, element ):
         if element.get( 'name' ):
             return True
         else:
             return False

    ##-------------------------------------------------------------------------------------------------
    def _dump_components( self ):
        
        for c in self.components:
            print '\ncomponent: {0.name}'.format( c )
            print '  variables:'
            for v in c.variables:
                print '    name: {0.name:<16}  public_interface: {0.public_interface:<4}  connection: {0.connection}'.format( v )

    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _get_containment_hierarchies( self ):
        
        ## <group/relationship_ref relationship="containment"> を探し、
        ## <componet> の階層構造を辞書 self.containment_hierarchies に格納する。
        ##
        ## { comp_1 : 
        ##     { comp_2 : 
        ##         { comp_3 : {},
        ##           comp_4 : {}
        ##         }
        ##       comp_5 : {},
        ##       comp_6 : {}
        ## }

        for gn in self.root_node.iterfind( './' + self.tag[ 'group' ] ):
            # gn for group node
            
            # groupエレメントの子に、relationship_refエレメントがなければエラー
            if gn.find( './' + self.tag[ 'relationship_ref' ] ) == None:
                raise TypeError, "<group> must have <relationship_ref> sub node."
            
            # <group relationship='containment'> に対する処理
            if gn.find( './' + self.tag[ 'relationship_ref' ] ).get( 'relationship' ) == 'containment':
                
                for top_level_cr in gn.iterfind( './' + self.tag[ 'component_ref' ] ):
                    
                    if top_level_cr.get( 'component' ) == None:
                        raise TypeError, "<component_ref> must have 'component' attribute."
                    
                    # print top_level_cr.get( 'component' )
                    self.containment_hierarchies[ str( top_level_cr.get( 'component' ) ) ] = \
                        self._get_component_ref_dict( top_level_cr )
        
        # <group>に含まれないcomponentをトップレベルに追加
        for component_name in [ c.name for c in self.components ]:
            if not ( self.exists_in_group( component_name ) ):
                self.containment_hierarchies[ component_name ] = {}
            

    ##-------------------------------------------------------------------------------------------------
    def _get_component_ref_dict( self, cr ):
        
        cr_dict = {}
        
        for child_cr in cr.iterfind( './' + self.tag[ 'component_ref' ] ):
            if child_cr.get( 'component' ) == None:
                raise TypeError, "<component_ref> must have 'component' attribute."
            
            cr_dict[ str( child_cr.get( 'component' ) ) ] = self._get_component_ref_dict( child_cr )
        
        return cr_dict

    ##-------------------------------------------------------------------------------------------------
    def exists_in_group( self, component_name ):
        
        for gn in self.root_node.iterfind( './' + self.tag[ 'group' ] ):
            if gn.find( './' + self.tag[ 'relationship_ref' ] ).get( 'relationship' ) == 'containment':
                for cr in gn.iterfind( './' + self.tag[ 'component_ref' ] ):
                     if component_name == cr.get( 'component' ):
                         return True
        return False

    ##-------------------------------------------------------------------------------------------------
    def _dump_containment_hierarchies( self ):
        
        depth = 0
        print '\n########################################################\nhierarchie:\n'
        
        for c, sub in self.containment_hierarchies.iteritems():
            print '{0}{1}'.format( ''.join( [ '  ' ] * depth ), c )
            
            self._dump_containment_hierarchies_recursive( sub, depth )

    ##-------------------------------------------------------------------------------------------------
    def _dump_containment_hierarchies_recursive( self, node, depth ):
        
        depth += 1
        
        for c, sub in node.iteritems():
            print '{0}{1}'.format( ''.join( [ '  ' ] * depth ), c )
            
            self._dump_containment_hierarchies_recursive( sub, depth )


    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _get_connections( self ):
        
        ## <connection> を走査し、同一関係にあるvariableの 
        ## variable_address のリストを self.connection に格納する。
        ##
        ## [ [ va_1, va_2,... ],... ]
        ##
        ## 他のvariableとconnectionを持つ場合、
        ##     self.components[ x ][ 'variable' ][ y ][ 'connection' ] = True
        ## に書き換える。
        
        for ce in self.root_node.iterfind( './' + self.tag[ 'connection' ] ):
            # ce for connection element
            
            map_components = ce.find( './' + self.tag[ 'map_components' ] )
            map_variables_iter  = ce.iterfind( './' + self.tag[ 'map_variables' ] )
            
            if None in ( map_components, map_variables_iter ):
                raise TypeError, "<connection> must have both of <map_components> and <map_variables> sub nodes."
            
            for map_variables in map_variables_iter:
                
                connection_pair = [ self.variable_address( map_components.get( 'component_1' ),
                                                           map_variables.get( 'variable_1' ) ),
                                    self.variable_address( map_components.get( 'component_2' ),
                                                           map_variables.get( 'variable_2' ) ) ]
                
                # DEBUG # print '########################################################\n_get_connections()\n  connection_pair: {0[0].component}:{0[0].name} & {0[1].component}:{0[1].name}'.format( connection_pair )
#               # DEBUG # self._dump_connections()
#               # DEBUG # raw_input( 'press any key' )
                
                icl = self._get_including_connection_list( connection_pair )
                if icl == None:
                    self.connections.append( connection_pair )
                    for va in connection_pair:
                        self._set_local_variable_connection_true( va )
                else:
                    for va in connection_pair:
                        exists = False
                        for x_va in icl:
                            if self._is_same_variable_address( va, x_va ):
                                exists = True
                        if not exists:
                            icl.append( va )
                            self._set_local_variable_connection_true( va )

    ##-------------------------------------------------------------------------------------------------
    def _get_including_connection_list( self, connection_pair ):
        
        ## connection_pair: 新たに読み取ったconnection（要素はva２つ）
        ## そのいずれかを含むconnection_listがself.connections中にあれば
        ## そのインデックスを返す。なければNoneを返す。
        
        for va in connection_pair:
            for x_cn in self.connections:
                for x_va in x_cn:
                    if self._is_same_variable_address( va, x_va ):
                        return x_cn
        return None

    ##-------------------------------------------------------------------------------------------------
    def _set_local_variable_connection_true( self, variable_address ):
        
        self._get_local_variable_by_variable_address( variable_address ).connection = True
        

    ##-------------------------------------------------------------------------------------------------
    def _is_same_variable_address( self, va_1, va_2 ):
        
        if ( str( va_1.component ) == str( va_2.component  ) ) and \
           ( str( va_1.name )      == str( va_2.name       ) ):
               return True
        else:
            return False

    ##-------------------------------------------------------------------------------------------------
    def _dump_connections( self ):
        
        print '\n########################################################\nconnection:\n'
        
        for cn in self.connections:
            # cn for connection list
            for va in cn:
                _indent = ''.join( [' '] * ( 60 - len( va.component ) - len( va.name ) ) )
                print '  {0.component}:{0.name}{1}{2}'.format( va, _indent, self._get_local_variable_by_variable_address( va ).public_interface )
            print ''


    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _get_global_variables( self ):
    # global_variablesを書き込む。
    # 引きつづき、global_mathsにglobal_mathを書き込む。
    
        for c in self.components:
            for v in c.variables:
                if not v.connection:
                    self.global_variables.append( self.global_variable(
                        v.name, c.name, v.initial_value, 
                        [ self.variable_address( c.name, v.name ) ] ) )
        
        for connection_list in self.connections:
            
            genuine_va = self._get_genuine_from_connection( connection_list )
            
            if genuine_va:
                self.global_variables.append( self.global_variable( 
                    genuine_va.name,
                    genuine_va.component,
                    self._get_local_variable_by_variable_address( genuine_va ).initial_value,
                    connection_list ) )
        
        for gm in self.global_maths:
            self._get_global_math( gm )


    ##-------------------------------------------------------------------------------------------------
    def _get_genuine_from_connection( self, connection_list ):
        
        ## connection_list中のvariable_addressから真正（genuine）を
        ## 選んで返す。
        ## CellMLの仕様では、public_interface = out のvariableは
        ## connection中にひとつしかなく、このvariableが初期値を持つことに
        ## なっているので、それを利用。
#        for va in connection_list:
#            print 'list   : {0.component}:{0.name}'.format( va )
#        print ''
        
        genuine = [ va for va in connection_list 
                    if self._get_local_variable_by_variable_address( va ).public_interface == 'out' ]
        
#        for gva in genuine:
#            print 'genuine: {0.component}:{0.name}'.format( gva )
#        print ''
        
        if len( genuine ) == 1:
            return genuine.pop()
        
        elif len( genuine ) >= 2:       #### FIXME #### public_interface == 'out'が２つ以上ある場合への対処
            return genuine.pop( 0 )
        
        else:
            raise TypeError, "Exact 1 variable among connection must be set public_interface = 'out'."

    ##-------------------------------------------------------------------------------------------------
    def _get_local_variable_by_variable_address( self, variable_address ):
        
        c = self._get_component_by_name( variable_address.component )
        
        for v in c.variables:
            if v.name == variable_address.name:
                return v
        
        return None

    ##-------------------------------------------------------------------------------------------------
    def _get_global_variable_by_variable_address( self, va ):
        
        # connectionに引数vaを持つglobal_variableを返す。
        # va: variable_addressオブジェクト
        
        for gv in self.global_variables:
            
            if ( gv.component == va.component ) and ( gv.name == va.name ):
#                print '        _get_global_variable_by_variable_address({0.component}:{0.name}) returns {1.component}:{1.name}'.format( va, gv )
                return gv
            
            for gv_va in gv.connection:
                if self._is_same_variable_address( va, gv_va ):
#                    print '        _get_global_variable_by_variable_address({0.component}:{0.name}) returns {1.component}:{1.name}'.format( va, gv )
                    return gv
        return None

    ##-------------------------------------------------------------------------------------------------
    def _get_component_by_name( self, name ):
        
        for e in self.components:
            if e.name == name:
                return e
        
        return None

    ##-------------------------------------------------------------------------------------------------
    def _get_global_math( self, gm ):
        
        if gm.math.root_node.tag == gm.math.tag[ 'ci' ]:
            ci_list = [ gm.math.root_node ]
        else:
            ci_list = gm.math.root_node.findall( './/' + gm.math.tag[ 'ci' ] )
        
        for ci in ci_list:
            _flag = False
#            print '\n  ci.text == {0.text}'.format( ci )
            for v in self._get_component_by_name( gm.component ).variables:
#                print '    v.name == {0.name}'.format( v )
#                raw_input( 'Press Any Key.' )
                if ci.text == v.name:
#                    print '    BINGO!'
                    _gv = self._get_global_variable_by_variable_address( self.variable_address( gm.component, v.name ) )
                    ci.text = '{0.component}:{0.name}'.format( _gv )
                    _flag = True
                    break
            
            if not _flag:
                raise TypeError, "gloval variable for [ {0}:{1} ] is not found.".format( gm.component, ci.text )
        
        gm.math.variable = gm.math.get_equation_variable()

    ##-------------------------------------------------------------------------------------------------
    def _dump_global_variables( self ):
        
        print '\n########################################################\nglobal_variables:\n'
        
        for gv in self.global_variables:
            _indent = ''.join( [' '] * ( 60 - len( gv.component ) - len( gv.name ) ) )
            print '  {0.component}:{0.name}{1}initial value = {0.initial_value}'.format( gv, _indent )
        print ''

    ##-------------------------------------------------------------------------------------------------
    def _dump_global_maths( self ):
        
        print '\n########################################################\nglobal_maths:\n'
        
        for gm in self.global_maths:
            print '  {0}\n'.format( gm.get_expression_str() )
        print ''


    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _calc_initial_values( self ):
        
        # 決定した初期値は該当するglobal_variablesに書き込む。
        # 新たに決定した初期値の数を返す。
        
        gain = 0
        prev_gain = -1
        round = 0
        
        while gain - prev_gain:
            
            round += 1
#            print '\n_calc_initial_values(): ROUND {0}\n'.format( round )
            
            prev_gain = gain
            
            for gv in self.global_variables:
                
                if not gv.has_initial_value():
                    
                    if not ( self._calc_initial_value( self.variable_address( gv.component, gv.name ) ) ):
                        # まず、実体に対して初期値計算を試みる。
                        # Falseの場合、connection関係にあるlocal_variableについても初期値計算を試みる
                        
                        for va in  [ _va for _va in gv.connection 
                                         if self._get_local_variable_by_variable_address( _va ).public_interface == 'out' ]:
                            
                            lv = self._get_local_variable_by_variable_address( va )
                            if lv.has_initial_value():
                                gv.initial_value = lv.initial_value
                                gain += 1
                                break
                            
                            else:
                                if self._calc_initial_value( va ):
                                    gain += 1
                                    break
                    else:
                        gain += 1
        
        return round

    ##-------------------------------------------------------------------------------------------------
    def _calc_initial_value( self, va ):
        
        # 新たな初期値を決定できた場合 True を、
        # できなかった場合 False を返す。
        
        c = self._get_component_by_name( va.component )
        
        m = [ _m for _m in c.maths if ( _m.type == CELLML_MATH_ASSIGNMENT_EQUATION ) and ( _m.variable == va.name ) ]
        
        if len( m ) >= 2:
            raise TypeError, "multiple maths describe variable '{0.name}' in component '{0.component}'.".format( va )
        
        elif len( m ) == 0:
            return False
        
        else:
            m = m[ 0 ]
            
            value = self._calc_math( m, m.right_side, va )
            
            if value == None:
                return False
            else:
                self._get_global_variable_by_variable_address( va ).initial_value = value
                return True

    ##-------------------------------------------------------------------------------------------------
    def _calc_math( self, math, element, va ):            ### MathML._convert_element_to_Expression()
        
        # math は 右辺 に限る。チェック機能は未実装。
        # 実数による値を求めて返す（float）。
        # 値を求められなかった場合、Noneを返す。
        
#        return 1.0
        
        _value = None
        
        if math.type != CELLML_MATH_ASSIGNMENT_EQUATION:
            raise TypeError, "_calc_math(): math.type must be assignment equation. ({0.component}:{0.name})".format( va )
        
        if   element.tag == math.tag[ 'cn' ]:
#            print '_calc_math( {0.component}:{0.name} )  <cn>'.format( va )
#            return float( element.text )
            _value = element.text
        
        elif element.tag == math.tag[ 'ci' ]:
#            print '_calc_math( {0.component}:{0.name} )  <ci>'.format( va )
            
            _value = self._get_global_variable_by_variable_address( self.variable_address( va.component, element.text ) ).initial_value
            
#            _gv = self._get_global_variable_by_variable_address( self.variable_address( va.component, element.text ) )
#            if _gv.has_initial_value():
##                print '    initial_value = {0.initial_value}'.format( _gv )
##                return float( _gv.initial_value )
#                print '    initial_value( {0.text} ) = {1}.'.format( element, _value )
#                _value = float( _gv.initial_value )
#            else:
#                print '    initial_value( {0.text} ) not found.'.format( element )
##                return None
#                pass
            
        elif element.tag == math.tag[ 'apply' ]:
#            print '_calc_math( {0.component}:{0.name} )  <apply>'.format( va )
#            return self._calc_math_apply( math, element, va )
            _value = self._calc_math_apply( math, element, va )
            
        elif element.tag == math.tag[ 'piecewise' ]:
#            print '_calc_math( {0.component}:{0.name} )  <piecewise>'.format( va )
#            return self._calc_math_piecewise( math, element, va )
            _value = self._calc_math_piecewise( math, element, va )
        
        else:
            raise TypeError, "_calc_math(): element tag '{0._get_tag_without_namespace(0.root_node.tag)}' is not implemented.".format( math )
        
#        if element == math.right_side:
#            print '_calc_math( {0.component}:{0.name} ) = {1}'.format( va, _value )
        
        if _value != None:
            _value = float( _value )
        
        return _value

    ##-------------------------------------------------------------------------------------------------
    def _calc_math_apply( self, math, element, va ):
        
#        return 1.0
        
        children = element.findall( './*' )
        tag = children.pop( 0 ).tag
       
        children_values = [ self._calc_math( math, child, va ) for child in children ]
        
        if None in children_values:
            return None
        
        # Unary arithmetic
        if ( tag == math.tag[ 'minus' ] ) and ( len( children_values ) == 1 ):
            return - children_values[ 0 ]
        
        # Binary arithmetic
        elif tag == math.tag[ 'divide' ]:
            return children_values[ 0 ] / children_values[ 1 ]
        
        elif tag == math.tag[ 'minus' ]:
            return children_values[ 0 ] - children_values[ 1 ]
        
        # Nary arithmetic
        elif tag in math.tag_group[ 'nary_arith' ]:
            _value = children_values.pop( 0 )
            
            while len( children_values ):
                
                if   tag == math.tag[ 'times' ]:
                    _value *= children_values.pop( 0 )
                elif tag == math.tag[ 'plus' ]:
                    _value += children_values.pop( 0 )
            
            return _value
        
        # Unary function
        elif ( tag in math.tag_group[ 'unary_func' ] ) or \
             ( tag in math.tag_group[ 'unary_binary_func' ] and len( children_values ) == 1 ):
            
            if tag == math.tag[ 'not' ]:
                return int( not( children_values[ 0 ] ) )
            
            elif tag == math.tag[ 'abs' ]:
                return abs( children_values[ 0 ] )
            
            elif tag == math.tag[ 'floor' ]:
                return floor( children_values[ 0 ] )
            
            elif tag == math.tag[ 'ceiling' ]:
                return ceil( children_values[ 0 ] )
            
            elif tag == math.tag[ 'factorial' ]:
                return factorial( children_values[ 0 ] )
            
            elif tag == math.tag[ 'exp' ]:
                return exp( children_values[ 0 ] )
            
            elif tag == math.tag[ 'ln' ]:
                return log( children_values[ 0 ] )
            
            elif tag == math.tag[ 'log' ]:
                return log10( children_values[ 0 ] )
            
            elif tag == math.tag[ 'sin' ]:
                return sin( children_values[ 0 ] )
            
            elif tag == math.tag[ 'cos' ]:
                return cos( children_values[ 0 ] )
            
            elif tag == math.tag[ 'tan' ]:
                return tan( children_values[ 0 ] )
            
            elif tag == math.tag[ 'arcsin' ]:
                return asin( children_values[ 0 ] )
            
            elif tag == math.tag[ 'arccos' ]:
                return acos( children_values[ 0 ] )
            
            elif tag == math.tag[ 'arctan' ]:
                return atan( children_values[ 0 ] )
            
            elif tag == math.tag[ 'root' ]:
                return sqrt( children_values[ 0 ] )

#            self.tag[ 'sec' ],
#            self.tag[ 'csc' ],
#            self.tag[ 'cot' ],
#            self.tag[ 'sinh' ],
#            self.tag[ 'cosh' ],
#            self.tag[ 'tanh' ],
#            self.tag[ 'sech' ],
#            self.tag[ 'csch' ],
#            self.tag[ 'coth' ],
#            self.tag[ 'arccosh' ],
#            self.tag[ 'arccot' ],
#            self.tag[ 'arccoth' ],
#            self.tag[ 'arccsc' ],
#            self.tag[ 'arccsch' ],
#            self.tag[ 'arcsec' ],
#            self.tag[ 'arcsech' ],
#            self.tag[ 'arcsinh' ],
#            self.tag[ 'arctanh' ], ],

        
        # Binary function
        
        elif tag == math.tag[ 'neq' ]:
            if children_values[ 0 ] != children_values[ 1 ]:
                return 1
            else:
                return 0
        
        elif tag == math.tag[ 'power' ]:
            return pow( children_values[ 0 ], children_values[ 1 ] )
        
        elif tag == math.tag[ 'root' ]:
            return pow( children_values[ 0 ], 1.0 / children_values[ 1 ] )
        
        # Nary nest function
        elif tag in math.tag_group[ 'nary_arith' ]:
            _value = children_values.pop( 0 )
            
            while len( children_values ):
                
                if   tag == math.tag[ 'and' ]:
                    _value = _value and children_values.pop( 0 )
                elif tag == math.tag[ 'or' ]:
                    _value = _value or children_values.pop( 0 )
                elif tag == math.tag[ 'xor' ]:
                    _value = not( _value or children_values.pop( 0 ) )
            
            return _value
        
        # Nary chain function
        elif tag in math.tag_group[ 'nary_chain_func' ]:
            
            _value   = True
            _value_2 = children_values.pop( 0 )
            
            while len( children_values ):
                
                _value_1 = _value_2
                _value_2 = children_values.pop( 0 )
                
                if   tag == math.tag[ 'eq' ]:
                    if not( _value_1 == _value_2 ):
                        _value = False
                elif tag == math.tag[ 'gt' ]:
                    if not( _value_1 > _value_2 ):
                        _value = False
                elif tag == math.tag[ 'lt' ]:
                    if not( _value_1 < _value_2 ):
                        _value = False
                elif tag == math.tag[ 'geq' ]:
                    if not( _value_1 >= _value_2 ):
                        _value = False
                elif tag == math.tag[ 'leq' ]:
                    if not( _value_1 <= _value_2 ):
                        _value = False
            
            if _value:
                return 1
            else:
                return 0
        
        else:
            raise TypeError, "_calc_math_apply(): tag '{0}' not cought in {1.component}:{1.name}".format( tag, va )
            return None


    ##-------------------------------------------------------------------------------------------------
    def _calc_math_piecewise( self, math, element, va ):
        
#        return 1.0
        
        sub_elements = element.findall( './*' )
        
        for sub_element in sub_elements:
            
            if sub_element.tag == math.tag[ 'piece' ]:
                
                children = sub_element.findall( './*' )
                
                if self._calc_math( math, children.pop(), va ):
                    return self._calc_math( math, children.pop(), va )
            
            elif sub_element.tag == math.tag[ 'otherwise' ]:
                child = sub_element.findall( './*' )
                return self._calc_math( math, child.pop(), va )
            
        return None


    ##-------------------------------------------------------------------------------------------------
    ##-------------------------------------------------------------------------------------------------
    def _divide_polynomial_ode( self ):
        
        print '\n########################################################\ndivided_odes:\n'
        
        for rate_eq in [ gm for gm in self.global_maths if gm.math.type == CELLML_MATH_RATE_EQUATION ]:
#            print '  {0}\n'.format( rate_eq.get_expression_str() )
            self._get_primary_terms( rate_eq )
        
        print ''
        
        for do in self.divided_odes:
            print '  {0:<16} : {1}\n'.format( do.variable, do.math.get_expression_str() )
        
        #self._integrate_terms()

    ##-------------------------------------------------------------------------------------------------
    def _get_primary_terms( self, gm ):
        
        # math : MathML オブジェクト
        # math（の右辺）が多項式なら真を返す。
        
        if gm.math.type in ( CELLML_MATH_ALGEBRAIC_EQUATION,
                             CELLML_MATH_ASSIGNMENT_EQUATION,
                             CELLML_MATH_RATE_EQUATION ):
            _element = gm.math.right_side
        else:
            _element = gm.math.root_node
        
        self.divided_odes.extend( 
            [ self.divided_ode( gm.component, MathML( _term, CELLML_MATH_NODE ), gm.math.variable ) 
                for _term in self._desolve_nested_polynomial( gm, _element ) ] )
        
        
    
    ##-------------------------------------------------------------------------------------------------
    def _desolve_nested_polynomial( self, gm, element, _sign = True ):
        
        # MathML の ElementTree 構造 element を解析し、多項式の場合には
        # 項に分解して 各項の ElementTree オブジェクトからなるリストを返す。
        
        _term_elements = []
        
        if element.tag == gm.tag[ 'apply' ]:
            children = element.findall( './*' )
            
            if children[ 0 ].tag == gm.tag[ 'plus' ]:
                children.pop( 0 )
                for child in children:
                    if child.tag == gm.tag[ 'apply' ]:
                        _term_elements.extend( self._desolve_nested_polynomial( gm, child, _sign ) )
                    else:
                        _term_elements.append( self._apply_sign( gm, child, _sign ) )
            
            elif children[ 0 ].tag == gm.tag[ 'minus' ] and len( children ) == 3:
                
                child = children.pop()    ## minus
                if child.tag == gm.tag[ 'apply' ]:
                    _term_elements.extend( self._desolve_nested_polynomial( gm, child, not _sign ) )
                else:
                    _term_elements.append( self._apply_sign( gm, child, not _sign ) )
                
                child = children.pop()    ## plus
                if child.tag == gm.tag[ 'apply' ]:
                    _term_elements.extend( self._desolve_nested_polynomial( gm, child, _sign ) )
                else:
                    _term_elements.append( self._apply_sign( gm, child, _sign ) )
            
            elif children[ 0 ].tag == gm.tag[ 'minus' ] and len( children ) == 2 :
                _term_elements.extend( self._desolve_nested_polynomial( gm, children.pop(), not _sign ) )
            
            else:
                _term_elements.append( self._apply_sign( gm, element, _sign ) )
        else:
            _term_elements.append( self._apply_sign( gm, element, _sign ) )
        
        return _term_elements

    ##-------------------------------------------------------------------------------------------------
    def _integrate_terms( self ):
        
        # _get_primary_terms() で生成された多項式リストを
        # スキャンし、同一項を括りだして量論行列を作成する。
        
        # 項全体に整数（あるいは、0.25の倍数）が含まれていたらcoefficientとして取りだす
        for do in self.divided_odes:
            if do.math.root_node.tag == do.tag[ 'apply' ]:
                children = do.math.root_node.findall( './*' )
                if children[ 0 ].tag == do.tag[ 'times' ] and \
                   children[ 1 ].tag == do.tag[ 'cn' ]:
                   if float( children[ 1 ].text ) == float( int( float( children[ 1 ].text ) * 4.0 ) ) / 4.0:
                       self.coefficient = float( children[ 1 ].text )
                       do.math.root_node.remove( children[ 1 ] )
        
        _unprocessed = [ do for do in self.divided_odes if do.genuine == 0 ]
        while len( _unprocessed ) > 0:
            if len( _unprocessed ) == 1:
                pass
                # 残り１つなら、単一のgenuine
            else:
                _subject = _unprocessed.pop()
                [ self._are_terms_identical( _subject, _object ) for _object in _unprocessed ]
                if _subject.genuine == 0:
                    _subject._set_stoichiometry( _subject.variable, _subject.coefficient )
                    _subject.coefficient = 1
                    _subject.genuine = 1
            
            _unprocessed = [ do for do in self.divided_odes if do.genuine == 0 ]


    ##-------------------------------------------------------------------------------------------------
    def _are_terms_identical( self, sub, ob ):
        
        # self.divided_odesをスキャンして、doと同一の項を探す。
        # 同一の項から「本物」を１つ決め genuine = 1 に設定し、量論行列を設定する。
        # その他の同一項は genuine = -1 に設定して検索対象から除外する。
        # スキャンし、同一項を括りだして量論行列を作成する。
        # do: divided_ode
        
        # 厳密には、plus または times の場合、その下層で plus / times がネストして
        # いたら解消すべきだが、現時点では未対応。
        
        # 数式全体に含まれるタグのリストをつくり、一致しなければFalseを返す
        _math_elements   = [ deepcopy( sub.math.root_node ), deepcopy( ob.math.root_node ) ]
        _all_elements_ls = [ _math_element.findall( './/*' ) for _math_element in _math_elements ].append( _math_element )  ## 自身も要素に加える
        _all_tags        = [ [ _element.tag for _element in _all_elements ] for _all_elements in _all_elements_ls ]
        
        if _all_tags[ 0 ].sort() != _all_tags[ 1 ].sort():
            return False
        
        return True
        

    ##-------------------------------------------------------------------------------------------------
    def _apply_sign( self, gm, element, _sign ):
        
        # MathMLのElementTree構造 element の符号を設定する。
        # bool型変数 _sign = True なら element をそのまま返す。
        # False なら 全体に -1 を乗じた element を返す。
        
        if _sign == False:
            _element = deepcopy( element )
            element.clear()
            element.tag = gm.tag[ 'apply' ]
            et.SubElement( element, gm.tag[ 'minus' ] )
            element.insert( 1, _element )
            
#            for e in element.findall( './/*' ).append( element ):
#                print '    {0}'.format( e.tag )
            
        return element



##=====================================================================================================
##  MathMLクラス
##=====================================================================================================

class MathML( object ):

    class Expression( object ):
        
        def __init__( self, string = '', priority = 0 ):
            self.string = string
            self.priority = priority

    def __init__( self, MathML, type = None ):
        
        self.root_node = MathML                   ## xml.etree.ElementTree.Element object
        
        # self.left_side  = self._get_left_side_Element()    ## 左辺のElementオブジェクト
        # self.right_side = self._get_right_side_Element()   ## 右辺のElementオブジェクト
        
        # CellML 1.0 で使用できる MathML タグ（https://www.cellml.org/specifications/cellml_1.0/index_html#sec_mathematics）
        self.tag = {
            
            'math' : '{{{0}}}math'.format( MATHML_NAMESPACE ),
            
          # token elements
            'cn' : '{{{0}}}cn'.format( MATHML_NAMESPACE ),
            'ci' : '{{{0}}}ci'.format( MATHML_NAMESPACE ),
            
          # basic content elements
            'apply'     : '{{{0}}}apply'.format( MATHML_NAMESPACE ),
            'piecewise' : '{{{0}}}piecewise'.format( MATHML_NAMESPACE ),
            'piece'     : '{{{0}}}piece'.format( MATHML_NAMESPACE ),
            'otherwise' : '{{{0}}}otherwise'.format( MATHML_NAMESPACE ),
            
          # relational operators
            'eq'  : '{{{0}}}eq'.format( MATHML_NAMESPACE ),
            'neq' : '{{{0}}}neq'.format( MATHML_NAMESPACE ),
            'gt'  : '{{{0}}}gt'.format( MATHML_NAMESPACE ),
            'lt'  : '{{{0}}}lt'.format( MATHML_NAMESPACE ),
            'geq' : '{{{0}}}geq'.format( MATHML_NAMESPACE ),
            'leq' : '{{{0}}}leq'.format( MATHML_NAMESPACE ),
            
          # arithmetic operators
            'plus'      : '{{{0}}}plus'.format( MATHML_NAMESPACE ),
            'minus'     : '{{{0}}}minus'.format( MATHML_NAMESPACE ),
            'times'     : '{{{0}}}times'.format( MATHML_NAMESPACE ),
            'divide'    : '{{{0}}}divide'.format( MATHML_NAMESPACE ),
            'power'     : '{{{0}}}power'.format( MATHML_NAMESPACE ),
            'root'      : '{{{0}}}root'.format( MATHML_NAMESPACE ),
            'abs'       : '{{{0}}}abs'.format( MATHML_NAMESPACE ),
            'exp'       : '{{{0}}}exp'.format( MATHML_NAMESPACE ),
            'ln'        : '{{{0}}}ln'.format( MATHML_NAMESPACE ),
            'log'       : '{{{0}}}log'.format( MATHML_NAMESPACE ),
            'floor'     : '{{{0}}}floor'.format( MATHML_NAMESPACE ),
            'ceiling'   : '{{{0}}}ceiling'.format( MATHML_NAMESPACE ),
            'factorial' : '{{{0}}}factorial'.format( MATHML_NAMESPACE ),
            'rem' : '{{{0}}}rem'.format( MATHML_NAMESPACE ),                ## not defined in official specification
            
          # logical operators
            'and' : '{{{0}}}and'.format( MATHML_NAMESPACE ),
            'or'  : '{{{0}}}or'.format( MATHML_NAMESPACE ),
            'xor' : '{{{0}}}xor'.format( MATHML_NAMESPACE ),
            'not' : '{{{0}}}not'.format( MATHML_NAMESPACE ),
            
          # calculus elements
            'diff' : '{{{0}}}diff'.format( MATHML_NAMESPACE ),
            
          # qualifier elements
            'degree'  : '{{{0}}}degree'.format( MATHML_NAMESPACE ),
            'bvar'    : '{{{0}}}bvar'.format( MATHML_NAMESPACE ),
            'logbase' : '{{{0}}}logbase'.format( MATHML_NAMESPACE ),
            
          # trigonometric operators
            'sin'     : '{{{0}}}sin'.format( MATHML_NAMESPACE ),
            'cos'     : '{{{0}}}cos'.format( MATHML_NAMESPACE ),
            'tan'     : '{{{0}}}tan'.format( MATHML_NAMESPACE ),
            'sec'     : '{{{0}}}sec'.format( MATHML_NAMESPACE ),
            'csc'     : '{{{0}}}csc'.format( MATHML_NAMESPACE ),
            'cot'     : '{{{0}}}cot'.format( MATHML_NAMESPACE ),
            'sinh'    : '{{{0}}}sinh'.format( MATHML_NAMESPACE ),
            'cosh'    : '{{{0}}}cosh'.format( MATHML_NAMESPACE ),
            'tanh'    : '{{{0}}}tanh'.format( MATHML_NAMESPACE ),
            'sech'    : '{{{0}}}sech'.format( MATHML_NAMESPACE ),
            'csch'    : '{{{0}}}csch'.format( MATHML_NAMESPACE ),
            'coth'    : '{{{0}}}coth'.format( MATHML_NAMESPACE ),
            'arcsin'  : '{{{0}}}arcsin'.format( MATHML_NAMESPACE ),
            'arccos'  : '{{{0}}}arccos'.format( MATHML_NAMESPACE ),
            'arctan'  : '{{{0}}}arctan'.format( MATHML_NAMESPACE ),
            'arccosh' : '{{{0}}}arccosh'.format( MATHML_NAMESPACE ),
            'arccot'  : '{{{0}}}arccot'.format( MATHML_NAMESPACE ),
            'arccoth' : '{{{0}}}arccoth'.format( MATHML_NAMESPACE ),
            'arccsc'  : '{{{0}}}arccsc'.format( MATHML_NAMESPACE ),
            'arccsch' : '{{{0}}}arccsch'.format( MATHML_NAMESPACE ),
            'arcsec'  : '{{{0}}}arcsec'.format( MATHML_NAMESPACE ),
            'arcsech' : '{{{0}}}arcsech'.format( MATHML_NAMESPACE ),
            'arcsinh' : '{{{0}}}arcsinh'.format( MATHML_NAMESPACE ),
            'arctanh' : '{{{0}}}arctanh'.format( MATHML_NAMESPACE ),
            
          # constants
            'true'         : '{{{0}}}true'.format( MATHML_NAMESPACE ),
            'false'        : '{{{0}}}false'.format( MATHML_NAMESPACE ),
            'notanumber'   : '{{{0}}}notanumber'.format( MATHML_NAMESPACE ),
            'pi'           : '{{{0}}}pi'.format( MATHML_NAMESPACE ),
            'infinity'     : '{{{0}}}infinity'.format( MATHML_NAMESPACE ),
            'exponentiale' : '{{{0}}}exponentiale'.format( MATHML_NAMESPACE ),
            
          # semantics and annotation elements
            'semantics'      : '{{{0}}}semantics'.format( MATHML_NAMESPACE ),
            'annotation'     : '{{{0}}}annotation'.format( MATHML_NAMESPACE ),
            'annotation-xml' : '{{{0}}}annotation-xml'.format( MATHML_NAMESPACE ),
            
        }
        
        
        ## ----------------------------------------
        ## MathMLのグループ  self.tag_group
        ## 反復処理のコードの簡略化のため
        ## ----------------------------------------
        # 
        self.tag_group = {

           # --------------------------------------------------------------------
           # 数式処理によるグループ
           # 未実装：degree, bvar, logbase, semantics, annotation, annotation-xml
           # --------------------------------------------------------------------
            'unary_func' : [ 
                self.tag[ 'not' ],
                
                self.tag[ 'abs' ],
                self.tag[ 'floor' ],
                self.tag[ 'ceiling' ],
                self.tag[ 'factorial' ],
                self.tag[ 'exp' ],
                self.tag[ 'ln' ],
                self.tag[ 'log' ],

                self.tag[ 'sin' ],
                self.tag[ 'cos' ],
                self.tag[ 'tan' ],
                self.tag[ 'sec' ],
                self.tag[ 'csc' ],
                self.tag[ 'cot' ],
                self.tag[ 'sinh' ],
                self.tag[ 'cosh' ],
                self.tag[ 'tanh' ],
                self.tag[ 'sech' ],
                self.tag[ 'csch' ],
                self.tag[ 'coth' ],
                self.tag[ 'arcsin' ],
                self.tag[ 'arccos' ],
                self.tag[ 'arctan' ],
                self.tag[ 'arccosh' ],
                self.tag[ 'arccot' ],
                self.tag[ 'arccoth' ],
                self.tag[ 'arccsc' ],
                self.tag[ 'arccsch' ],
                self.tag[ 'arcsec' ],
                self.tag[ 'arcsech' ],
                self.tag[ 'arcsinh' ],
                self.tag[ 'arctanh' ], ],

            'unary_binary_func' : [ 
                self.tag[ 'root' ], ],

            'binary_func' : [ 
                self.tag[ 'neq' ],
                self.tag[ 'power' ],
                self.tag[ 'rem' ], ],

            'nary_chain_func' : [ 
                self.tag[ 'eq' ],
                self.tag[ 'gt' ],
                self.tag[ 'lt' ],
                self.tag[ 'geq' ],
                self.tag[ 'leq' ], ],

            'nary_nest_func' : [ 
                self.tag[ 'and' ],
                self.tag[ 'or' ],
                self.tag[ 'xor' ], ],

            'unary_binary_arith' : [ 
                self.tag[ 'minus' ], ],

            'binary_arith' : [ 
                self.tag[ 'divide' ], ],

            'nary_arith' : [ 
                self.tag[ 'times' ],
                self.tag[ 'plus' ], ],

            'piecewise' : [ 
                self.tag[ 'piecewise' ],
                self.tag[ 'piece' ],
                self.tag[ 'otherwise' ] ],

           # --------------------------------------------------------------------
           # タグの性格による分類
           # --------------------------------------------------------------------
            'token' : [ 
                self.tag[ 'cn' ],           # 数字
                self.tag[ 'ci' ] ],         # 識別子

            'basic_content' : [ 
                self.tag[ 'apply' ],        # 演算子適用宣言（直後には演算子）
                self.tag[ 'piecewise' ],
                self.tag[ 'piece' ],
                self.tag[ 'otherwise' ] ],

            'relational_operators' : [ 
                self.tag[ 'eq' ],           # 等号と比較演算子の区別はどうつけているのだろう？？？
                self.tag[ 'neq' ],          
                self.tag[ 'gt' ],
                self.tag[ 'lt' ],
                self.tag[ 'geq' ],
                self.tag[ 'leq' ] ],

            'arithmetic_operators' : [ 
                self.tag[ 'plus' ],
                self.tag[ 'minus' ],
                self.tag[ 'times' ],
                self.tag[ 'divide' ],
                self.tag[ 'power' ],
                self.tag[ 'root' ],
                self.tag[ 'abs' ],
                self.tag[ 'exp' ],
                self.tag[ 'ln' ],
                self.tag[ 'log' ],
                self.tag[ 'floor' ],
                self.tag[ 'ceiling' ],
                self.tag[ 'factorial' ],
                self.tag[ 'rem' ] ],

            'logical_operators' : [ 
                self.tag[ 'and' ],
                self.tag[ 'or' ],
                self.tag[ 'xor' ],
                self.tag[ 'not' ] ],

            'calculus' : [ 
                self.tag[ 'diff' ] ],       # 微分演算子

            'qualifier' : [ 
                self.tag[ 'degree' ],       # 次数
                self.tag[ 'bvar' ],         # 変数の結合（dv/dtなど）
                self.tag[ 'logbase' ] ],    # 対数の底

            'trigonometric_operators' : [ 
                self.tag[ 'sin' ],
                self.tag[ 'cos' ],
                self.tag[ 'tan' ],
                self.tag[ 'sec' ],
                self.tag[ 'csc' ],
                self.tag[ 'cot' ],
                self.tag[ 'sinh' ],
                self.tag[ 'cosh' ],
                self.tag[ 'tanh' ],
                self.tag[ 'sech' ],
                self.tag[ 'csch' ],
                self.tag[ 'coth' ],
                self.tag[ 'arcsin' ],
                self.tag[ 'arccos' ],
                self.tag[ 'arctan' ],
                self.tag[ 'arccosh' ],
                self.tag[ 'arccot' ],
                self.tag[ 'arccoth' ],
                self.tag[ 'arccsc' ],
                self.tag[ 'arccsch' ],
                self.tag[ 'arcsec' ],
                self.tag[ 'arcsech' ],
                self.tag[ 'arcsinh' ],
                self.tag[ 'arctanh' ] ],

            'constants' : [ 
                self.tag[ 'true' ],
                self.tag[ 'false' ],
                self.tag[ 'notanumber' ],
                self.tag[ 'pi' ],
                self.tag[ 'infinity' ],
                self.tag[ 'exponentiale' ] ],

            'semantics_annotation' : [ 
                self.tag[ 'semantics' ],
                self.tag[ 'annotation' ],
                self.tag[ 'annotation-xml' ] ]
        }
        
        
        ## ------------------------------------------------
        ## 演算子のExpression属性中での表現 self.operator_str
        ## ------------------------------------------------
        self.operator_str = {
          # basic content elements
            self.tag[ 'piecewise' ] : 'piecewise',
            # self.tag[ 'piece' ] : 'piece',
            # self.tag[ 'otherwise' ] : 'otherwise',
            
          # relational operators
            self.tag[ 'eq' ]  : 'eq',
            self.tag[ 'neq' ] : 'neq',
            self.tag[ 'gt' ]  : 'gt',
            self.tag[ 'lt' ]  : 'lt',
            self.tag[ 'geq' ] : 'geq',
            self.tag[ 'leq' ] : 'leq',
            
          # arithmetic operators
            self.tag[ 'plus' ]      : '+',
            self.tag[ 'minus' ]     : '-',
            self.tag[ 'times' ]     : '*',
            self.tag[ 'divide' ]    : '/',
            self.tag[ 'power' ]     : 'pow',
            self.tag[ 'root' ]      : 'sqrt',  # when Unary
            self.tag[ 'abs' ]       : 'abs',
            self.tag[ 'exp' ]       : 'exp',
            self.tag[ 'ln' ]        : 'log',
            self.tag[ 'log' ]       : 'log10',
            self.tag[ 'floor' ]     : 'floor',
            self.tag[ 'ceiling' ]   : 'ceil',
            self.tag[ 'factorial' ] : 'factorial',
            self.tag[ 'rem' ] : 'rem',
            
          # logical operators
            self.tag[ 'and' ] : 'and',
            self.tag[ 'or' ]  : 'or',
            self.tag[ 'xor' ] : 'xor',
            self.tag[ 'not' ] : 'not',
            
          # calculus elements
            self.tag[ 'diff' ] : 'diff',
            
          # qualifier elements
            self.tag[ 'degree' ]  : 'degree',
            self.tag[ 'bvar' ]    : 'bvar',
            self.tag[ 'logbase' ] : 'logbase',
            
          # trigonometric operators
            self.tag[ 'sin' ]     : 'sin',
            self.tag[ 'cos' ]     : 'cos',
            self.tag[ 'tan' ]     : 'tan',
            self.tag[ 'sec' ]     : 'sec',
            self.tag[ 'csc' ]     : 'csc',
            self.tag[ 'cot' ]     : 'cot',
            self.tag[ 'sinh' ]    : 'sinh',
            self.tag[ 'cosh' ]    : 'cosh',
            self.tag[ 'tanh' ]    : 'tanh',
            self.tag[ 'sech' ]    : 'sech',
            self.tag[ 'csch' ]    : 'csch',
            self.tag[ 'coth' ]    : 'coth',
            self.tag[ 'arcsin' ]  : 'arcsin',
            self.tag[ 'arccos' ]  : 'arccos',
            self.tag[ 'arctan' ]  : 'arctan',
            self.tag[ 'arccosh' ] : 'arccosh',
            self.tag[ 'arccot' ]  : 'arccot',
            self.tag[ 'arccoth' ] : 'arccoth',
            self.tag[ 'arccsc' ]  : 'arccsc',
            self.tag[ 'arccsch' ] : 'arccsch',
            self.tag[ 'arcsec' ]  : 'arcsec',
            self.tag[ 'arcsech' ] : 'arcsech',
            self.tag[ 'arcsinh' ] : 'arcsinh',
            self.tag[ 'arctanh' ] : 'arctanh',
            
          # constants
            self.tag[ 'true' ]         : 'true',
            self.tag[ 'false' ]        : 'false',
            self.tag[ 'notanumber' ]   : 'notanumber',
            self.tag[ 'pi' ]           : 'pi',
            self.tag[ 'infinity' ]     : 'infinity',
            self.tag[ 'exponentiale' ] : 'exponentiale',
            
          # semantics and annotation elements
            self.tag[ 'semantics' ]      : 'semantics',
            self.tag[ 'annotation' ]     : 'annotation',
            self.tag[ 'annotation-xml' ] : 'annotation-xml',
        }
        
        
        ## --------------------------------------
        ## 演算子の優先順位 self.operator_priority
        ## --------------------------------------
        self.operator_priority = {
          # token elements
            self.tag[ 'cn' ] : 8,
            self.tag[ 'ci' ] : 8,
            
          # basic content elements
            self.tag[ 'apply' ]     : 0,
            self.tag[ 'piecewise' ] : 8,
            self.tag[ 'piece' ]     : 0,
            self.tag[ 'otherwise' ] : 0,
            
          # relational operators
            self.tag[ 'eq' ]  : 0,
            self.tag[ 'neq' ] : 0,
            self.tag[ 'gt' ]  : 0,
            self.tag[ 'lt' ]  : 0,
            self.tag[ 'geq' ] : 0,
            self.tag[ 'leq' ] : 0,
            
          # arithmetic operators
            self.tag[ 'plus' ]      : 2,
            self.tag[ 'minus' ]     : 2,
            self.tag[ 'times' ]     : 4,
            self.tag[ 'divide' ]    : 4,
            self.tag[ 'power' ]     : 8,
            self.tag[ 'root' ]      : 8,
            self.tag[ 'abs' ]       : 8,
            self.tag[ 'exp' ]       : 8,
            self.tag[ 'ln' ]        : 8,
            self.tag[ 'log' ]       : 8,
            self.tag[ 'floor' ]     : 8,
            self.tag[ 'ceiling' ]   : 8,
            self.tag[ 'factorial' ] : 8,
            self.tag[ 'rem' ]       : 8,
            
          # logical operators
            self.tag[ 'and' ] : 8,
            self.tag[ 'or' ]  : 8,
            self.tag[ 'xor' ] : 8,
            self.tag[ 'not' ] : 8,
            
          # calculus elements
            self.tag[ 'diff' ] : 8,
            
          # qualifier elements
            self.tag[ 'degree' ]  : 0,
            self.tag[ 'bvar' ]    : 0,
            self.tag[ 'logbase' ] : 0,
            
          # trigonometric operators
            self.tag[ 'sin' ]     : 0,
            self.tag[ 'cos' ]     : 0,
            self.tag[ 'tan' ]     : 0,
            self.tag[ 'sec' ]     : 0,
            self.tag[ 'csc' ]     : 0,
            self.tag[ 'cot' ]     : 0,
            self.tag[ 'sinh' ]    : 0,
            self.tag[ 'cosh' ]    : 0,
            self.tag[ 'tanh' ]    : 0,
            self.tag[ 'sech' ]    : 0,
            self.tag[ 'csch' ]    : 0,
            self.tag[ 'coth' ]    : 0,
            self.tag[ 'arcsin' ]  : 0,
            self.tag[ 'arccos' ]  : 0,
            self.tag[ 'arctan' ]  : 0,
            self.tag[ 'arccosh' ] : 0,
            self.tag[ 'arccot' ]  : 0,
            self.tag[ 'arccoth' ] : 0,
            self.tag[ 'arccsc' ]  : 0,
            self.tag[ 'arccsch' ] : 0,
            self.tag[ 'arcsec' ]  : 0,
            self.tag[ 'arcsech' ] : 0,
            self.tag[ 'arcsinh' ] : 0,
            self.tag[ 'arctanh' ] : 0,
            
          # constants
            self.tag[ 'true' ]         : 0,
            self.tag[ 'false' ]        : 0,
            self.tag[ 'notanumber' ]   : 0,
            self.tag[ 'pi' ]           : 0,
            self.tag[ 'infinity' ]     : 0,
            self.tag[ 'exponentiale' ] : 0,
            
          # semantics and annotation elements
            self.tag[ 'semantics' ]      : 0,
            self.tag[ 'annotation' ]     : 0,
            self.tag[ 'annotation-xml' ] : 0,
        }
        
        
        ## ------------------------------------------
        ## 左辺の形式ごとのタグ構造のパターン
        ## 階層化せず、ベタに上から下へ読んだ場合のパターン
        ## ------------------------------------------
        self.tag_pattern = {
            CELLML_MATH_ASSIGNMENT_EQUATION : [ [ self.tag[ 'ci' ] ] ],
            CELLML_MATH_RATE_EQUATION       : [ [ self.tag[ 'apply' ], self.tag[ 'diff' ], self.tag[ 'bvar' ], self.tag[ 'ci' ], self.tag[ 'ci' ] ] ]
        }
    
        ## ------------------------------------------
        ## 方程式の型、従属変数
        ## ------------------------------------------
        if type:
            self.type = type
            self.variable = None
        else:
            self.type = self.get_equation_type()      ## 方程式の型。以下の定数のいずれかを持つ
            self.variable = self.get_equation_variable()
            self.right_side = self._get_right_side_Element()   ## 右辺のElement
            self.string = self.get_right_side().get_expression_str()    ## 右辺の式を文字列にしたもの→Expressionとして使うテンプレート
    
    ##-------------------------------------------------------------------------------------------------
    ## 左右の辺、方程式の型、従属変数の取得メソッド
    ##-------------------------------------------------------------------------------------------------
    def get_equation_type( self ):
        
        left_side_Element = self._get_left_side_Element()
        
        if left_side_Element == None:
            raise TypeError, "Left side of equation is not found."
        
        tags = []
        
        for element in left_side_Element.iter():
            
            tags.append( element.tag )
        
        for type, tag_pattern in self.tag_pattern.iteritems():
            if tags in tag_pattern:
                return type
        
        return None
    
    ##-------------------------------------------------------------------------------------------------
    def get_equation_variable( self ):
        
        left_side_Element = self._get_left_side_Element()
        
        if left_side_Element == None:
            raise TypeError, "Left side of equation is not found."
        
        if left_side_Element.tag == self.tag[ 'apply' ]:  # differential equasion
            return left_side_Element.findall( './*' ).pop().text
        else:
            return left_side_Element.text
        
        return None
    
    ##-------------------------------------------------------------------------------------------------
    def get_left_side( self ):
        return MathML( self._get_left_side_Element(), CELLML_MATH_LEFT_SIDE )
    
    ##-------------------------------------------------------------------------------------------------
    def _get_left_side_Element( self ):
        
        find_eq = False
        count = 0
        
        for element in self.root_node.iter():
            
            # print '\n_get_left_side_Element() type: %s\n' % type( element )
            
            if count == 0 and element.tag != self.tag[ 'apply' ]:
                return None
            
            elif count == 1 and element.tag != self.tag[ 'eq' ]:
                return None
            
            elif count == 2:
                if element.tag in ( self.tag[ 'apply' ], self.tag[ 'ci' ] ):
                    return element
                else:
                    return None
            
            else:
                count += 1
        
        return None
        
    ##-------------------------------------------------------------------------------------------------
    def get_right_side( self ):
        return MathML( self._get_right_side_Element(), CELLML_MATH_RIGHT_SIDE )
    
    ##-------------------------------------------------------------------------------------------------
    def _get_right_side_Element( self ):
        
        find_eq = False
        find_left_side = False
        
        for element in self.root_node.iterfind( './*' ):
            
            if element.tag == self.tag[ 'eq' ]:
                find_eq = True
            
            elif find_eq and ( not find_left_side ):
                find_left_side = True
            
            elif find_eq and find_left_side:
                
                return element
        
        return None
    
    ##-------------------------------------------------------------------------------------------------
    ## Elememtを文字列に変換するためのメソッド
    ##-------------------------------------------------------------------------------------------------
    def get_expression_str( self ):
        
        if self.type in ( CELLML_MATH_ALGEBRAIC_EQUATION,
                          CELLML_MATH_ASSIGNMENT_EQUATION,
                          CELLML_MATH_RATE_EQUATION ):
            
            return '%s = %s' % ( self.get_left_side().get_expression_str(),
                                 self.get_right_side().get_expression_str() )
        else:
            return self._convert_element_to_Expression( self.root_node ).string
    
    ##-------------------------------------------------------------------------------------------------
    def _convert_element_to_Expression( self, element ):
        
        if element.tag in self.tag_group[ 'token' ]:
            return self.Expression( element.text, self.operator_priority[ element.tag ] )
            
        elif element.tag == self.tag[ 'apply' ]:
            return self._convert_apply_element_to_Expression( element )
            
        elif element.tag == self.tag[ 'piecewise' ]:
            return self._convert_piecewise_element_to_Expression( element )
        
        else:
            return self.Expression( '((( %s,... )))' % self._get_tag_without_namespace( element.tag ), 255 )
    
    ##-------------------------------------------------------------------------------------------------
    def _convert_apply_element_to_Expression( self, element ):
        
        children = element.findall( './*' )
        
        operator = children.pop( 0 )
        return self._convert_applying_Elements_to_Expression( operator.tag, children )
    
    ##-------------------------------------------------------------------------------------------------
    def _convert_piecewise_element_to_Expression( self, element ):
        
        #pieces = element.findall( './piece' )
        #otherwise = element.findall( './otherwise' )
        #if len( otherwise ) > 1:
        #    raise TypeError, '<piecewise> element must have 0 or 1 <otherwise> child.'
        #
        #print 'piecewise: piece( %i ), otherwise( %i ), (total %i Elements)' % ( len( pieces ), len( otherwise ), len( element.findall( './*' ) ) )
        
        sub_elements = element.findall( './*' )
        
        piece_Expressions = []
        otherwise_Expression = None
        
        for sub_element in sub_elements:
            
            if sub_element.tag == self.tag[ 'piece' ]:
                
                children = sub_element.findall( './*' )
                
                if len( children ) != 2:
                    raise TypeError, '<piece> element must have exactly 2 children.'
                
                piece_Expressions.append( 
                    dict(
                        condition = self._convert_element_to_Expression( children.pop() ),
                        value     = self._convert_element_to_Expression( children.pop() )
                        )
                    )
            
            elif sub_element.tag == self.tag[ 'otherwise' ]:
                
                if otherwise_Expression:
                    raise TypeError, '<piecewise> element must have 0 or 1 <otherwise> child.'
                
                child = sub_element.findall( './*' )
                
                if len( child ) != 1:
                    raise TypeError, '<otherwise> element must have exactly 1 child.'
                
                otherwise_Expression = self._convert_element_to_Expression( child.pop() )
            
            else:
                raise TypeError, '<piecewise> element\'s child must be <piece> or <otherwise> element.'
            
        return self._arrange_piecewise_Expression( piece_Expressions, otherwise_Expression )
    
    ##-------------------------------------------------------------------------------------------------
    def _convert_applying_Elements_to_Expression( self, tag, children ):
       
        operator = None
       
        children_Expressions = []
       
        for child in children:
            children_Expressions.append( self._convert_element_to_Expression( child ) )
        
        # Unary arithmetic
        if tag  in self.tag_group[ 'unary_binary_arith' ] and len( children_Expressions ) == 1:
            
            return self.Expression( self.operator_str[ tag ] + self._get_parenthesized_expression_string( 8, children_Expressions[ 0 ] ), 8 )
        
        # Binary arithmetic
        elif tag == self.tag[ 'divide' ]:
            if len( children_Expressions ) != 2:
                raise TypeError, 'Operator "%s" must have exactly 2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = ( 
                self._get_parenthesized_expression_string( 4, children_Expressions[ 0 ] ), 
#                children_Expressions[ 0 ].string,
                self._get_parenthesized_expression_string( 8, children_Expressions[ 1 ] ) )
            operator = ' %s ' % self.operator_str[ tag ]
            return self.Expression( operator.join( children_expression_strings ), self.operator_priority[ tag ] )
        
        elif tag == self.tag[ 'minus' ]:
            if len( children_Expressions ) != 2:
                raise TypeError, 'Operator "%s" must have exactly 2 children.' % self._get_tag_without_namespace( tag )
            
#            children_expression_strings = self._get_parenthesized_expression_strings( self.operator_priority[ tag ], children_Expressions )
            children_expression_strings = ( 
                self._get_parenthesized_expression_string( 2, children_Expressions[ 0 ] ), 
                self._get_parenthesized_expression_string( 3, children_Expressions[ 1 ] ) )  # 後半は加算を括弧でくくる必要あり
            operator = ' %s ' % self.operator_str[ tag ]
            return self.Expression( operator.join( children_expression_strings ), self.operator_priority[ tag ] )
        
        # Nary arithmetic
        elif tag in self.tag_group[ 'nary_arith' ]:
            if len( children_Expressions ) < 2:
                raise TypeError, 'Operator "%s" must have >= 2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = self._get_parenthesized_expression_strings( self.operator_priority[ tag ], children_Expressions )
            operator = ' %s ' % self.operator_str[ tag ]
            return self.Expression( operator.join( children_expression_strings ), self.operator_priority[ tag ] )
        
        # Unary function
        elif ( tag in self.tag_group[ 'unary_func' ] ) or \
             ( tag in self.tag_group[ 'unary_binary_func' ] and len( children_Expressions ) == 1 ):
            if len( children_Expressions ) != 1:
                raise TypeError, 'Operator "%s" must have exactly 1 child.' % self._get_tag_without_namespace( tag )
            
            return self.Expression( '%s( %s )' % ( self.operator_str[ tag ], 
                                                   children_Expressions[ 0 ].string ), 
                                    self.operator_priority[ tag ] )
        
        # Binary function
        elif tag in self.tag_group[ 'binary_func' ]:
            if len( children_Expressions ) != 2:
                raise TypeError, 'Operator "%s" must have exactly 2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = self._get_parenthesized_expression_strings( 0, children_Expressions )
            
            format_strings = [ self.operator_str[ tag ] ]
            format_strings.extend( children_expression_strings )
            # print '\n_convert_applying_Elements_to_Expression::format_strings = %s\n' % format_strings
            return self.Expression( '%s( %s, %s )' % tuple( format_strings ), self.operator_priority[ tag ] )
        
        # Binary function - root
        elif tag == self.tag[ 'root' ]:
            if len( children_Expressions ) != 2:
                raise TypeError, 'Operator "%s" must have exactly 1 or 2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = self._get_parenthesized_expression_strings( 0, children_Expressions )
            
            format_strings = ( 'pow', 
                               children_expression_strings[ 0 ], 
                               '1 / %s' % self._get_parenthesized_expression_string( 8, children_expression_strings[ 1 ] ) )
            return self.Expression( '%s( %s, %s )' % format_strings, self.operator_priority[ tag ] )
        
        # Nary nest function
        elif tag in self.tag_group[ 'nary_nest_func' ]:
            if len( children_Expressions ) < 2:
                raise TypeError, 'Operator "%s" must have >=2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = self._get_parenthesized_expression_strings( 0, children_Expressions )
            
            return_string = '%s( %s, %s )' % ( self.operator_str[ tag ], 
                                               children_expression_strings.pop( 0 ), 
                                               children_expression_strings.pop( 0 ) )
            
            while len( children_expression_strings ):
                
                return_string = '%s( %s, %s )' % ( self.operator_str[ tag ], 
                                                   return_string, 
                                                   children_expression_strings.pop( 0 ) )
            
            return self.Expression( return_string, self.operator_priority[ tag ] )
        
        # Nary chain function
        elif tag in self.tag_group[ 'nary_chain_func' ]:
            if len( children_Expressions ) < 2:
                raise TypeError, 'Operator "%s" must have >=2 children.' % self._get_tag_without_namespace( tag )
            
            children_expression_strings = self._get_parenthesized_expression_strings( 0, children_Expressions )
            
            child_1 = children_expression_strings.pop( 0 )
            child_2 = children_expression_strings.pop( 0 )
            
            chain = []
            
            chain.append( '%s( %s, %s )' % ( self.operator_str[ tag ], child_1, child_2 ) )
            
            while len( children_expression_strings ):
                child_1 = child_2
                child_2 = children_expression_strings.pop( 0 )
                chain.append( '%s( %s, %s )' % ( self.operator_str[ tag ], child_1, child_2 ) )
            
            if len( chain ) == 1:
                return self.Expression( chain.pop(), self.operator_priority[ tag ] )
            
            else:
                chained_string = 'and( %s, %s )' % ( chain.pop( 0 ), chain.pop( 0 ) )
                
                while len( chain ):
                    chained_string = 'and( %s, %s )' % ( chained_string, chain.pop( 0 ) )
                
                return elf.Expression( chained_string, self.operator_priority[ self.tag[ 'and' ] ] )
        
        # diff
        elif tag == self.tag[ 'diff' ]:
            if len( children_Expressions ) != 2:
                raise TypeError, '"diff" element must have exactly 2 children.'
            
            ci   = children.pop().text
            bvar = children.pop()
            
            if bvar.findall( './*' ).pop().text == 'time':
                return self.Expression( 'd(%s)/dt' % ci, self.operator_priority[ tag ] )
            else:
                return self.Expression( 'd(%s)/d(%s)' % ( ci, bvar.findall( './*' ).pop().text ), self.operator_priority[ tag ] )
        
        return self.Expression( '((( %s,... )))' % self._get_tag_without_namespace( tag ), 255 )
    
    ##-------------------------------------------------------------------------------------------------
    def _get_parenthesized_expression_strings( self, operator_priorty, Expressions ):
        
        parenthesized_expression_strings = []
        
        for Expression in Expressions:
            parenthesized_expression_strings.append( self._get_parenthesized_expression_string( operator_priorty, Expression ) )
        
        return parenthesized_expression_strings
    
    ##-------------------------------------------------------------------------------------------------
    def _get_parenthesized_expression_string( self, operator_priorty, Expression ):
        
        if operator_priorty > Expression.priority:
            return '( %s )' % Expression.string
        else:
            return Expression.string
    
    ##-------------------------------------------------------------------------------------------------
    def _arrange_piecewise_Expression( self, piece_Expressions, otherwise_Expression ):
        
        args = []
        
        for piece in piece_Expressions:
            args.append( piece[ 'value' ].string )
            args.append( piece[ 'condition' ].string )
        
        if otherwise_Expression:
            args.append( otherwise_Expression.string )
        
        args = ', '.join( args )
        
        return self.Expression( '%s( %s )' % ( self.operator_str[ self.tag[ 'piecewise' ] ], args ), 
                                self.operator_priority[ self.tag[ 'piecewise' ] ] )
    
    ##-------------------------------------------------------------------------------------------------
    def _get_tag_without_namespace( self, tag ):
        return tag.split( '}' ).pop()
