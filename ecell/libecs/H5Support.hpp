#ifndef H5SUPPORT_HPP
#define H5SUPPORT_HPP

#include <H5Cpp.h>
#include <Packer.hpp>
#include <utility>
#include <string>
#include <cstring>
#include <cstddef>
#include <boost/assert.hpp>
#include <boost/variant.hpp>
#include <boost/optional.hpp>
#include <boost/array.hpp>
#include <boost/scoped_array.hpp>

template<typename T>
struct get_h5_scalar_data_type_le
{
    void operator()() const
    {
    }
};

template<typename T, std::size_t N>
struct get_h5_scalar_data_type_le<T[N]>
{
    H5::ArrayType operator()() const
    {
        static const hsize_t dims[] = { N };
        return H5::ArrayType(get_h5_scalar_data_type_le<T>()(), 1, dims);
    }
};

template<typename T, std::size_t N>
struct get_h5_scalar_data_type_le<boost::array<T, N> >
{
    H5::ArrayType operator()() const
    {
        static const hsize_t dims[] = { N };
        return H5::ArrayType(get_h5_scalar_data_type_le<T>()(), 1, dims);
    }
};

template<std::size_t N>
struct get_h5_scalar_data_type_le<char[N]>
{
    H5::StrType operator()() const
    {
        return H5::StrType(H5::PredType::C_S1, N);
    }
};

#define MAP_H5_SCALAR_TYPE_LE(type, h5type) \
    template<> struct get_h5_scalar_data_type_le<type> \
    { H5::DataType operator()() const { return h5type; } }

MAP_H5_SCALAR_TYPE_LE(char, H5::PredType::C_S1);
MAP_H5_SCALAR_TYPE_LE(uint8_t, H5::PredType::STD_U8LE);
MAP_H5_SCALAR_TYPE_LE(uint16_t, H5::PredType::STD_U16LE);
MAP_H5_SCALAR_TYPE_LE(uint32_t, H5::PredType::STD_U32LE);
MAP_H5_SCALAR_TYPE_LE(uint64_t, H5::PredType::STD_U64LE);
MAP_H5_SCALAR_TYPE_LE(int8_t, H5::PredType::STD_I8LE);
MAP_H5_SCALAR_TYPE_LE(int16_t, H5::PredType::STD_I16LE);
MAP_H5_SCALAR_TYPE_LE(int32_t, H5::PredType::STD_I32LE);
MAP_H5_SCALAR_TYPE_LE(int64_t, H5::PredType::STD_I64LE);
MAP_H5_SCALAR_TYPE_LE(float, H5::PredType::IEEE_F32LE);
MAP_H5_SCALAR_TYPE_LE(double, H5::PredType::IEEE_F64LE);
MAP_H5_SCALAR_TYPE_LE(long double, H5::PredType::NATIVE_LDOUBLE);

#undef MAP_H5_SCALAR_TYPE_LE

template<typename T>
struct get_h5_scalar_data_type_be
{
    void operator()() const
    {
    }
};

template<typename T, std::size_t N>
struct get_h5_scalar_data_type_be<T[N]>
{
    H5::ArrayType operator()() const
    {
        static const hsize_t dims[] = { N };
        return H5::ArrayType(get_h5_scalar_data_type_be<T>()(), 1, dims);
    }
};

template<typename T, std::size_t N>
struct get_h5_scalar_data_type_be<boost::array<T, N> >
{
    H5::ArrayType operator()() const
    {
        static const hsize_t dims[] = { N };
        return H5::ArrayType(get_h5_scalar_data_type_be<T>()(), 1, dims);
    }
};

template<std::size_t N>
struct get_h5_scalar_data_type_be<char[N]>
{
    H5::StrType operator()() const
    {
        return H5::StrType(H5::PredType::C_S1, N);
    }
};

#define MAP_H5_SCALAR_TYPE_BE(type, h5type) \
    template<> struct get_h5_scalar_data_type_be<type> \
    { H5::DataType operator()() const { return h5type; } }

MAP_H5_SCALAR_TYPE_BE(char, H5::PredType::C_S1);
MAP_H5_SCALAR_TYPE_BE(uint8_t, H5::PredType::STD_U8BE);
MAP_H5_SCALAR_TYPE_BE(uint16_t, H5::PredType::STD_U16BE);
MAP_H5_SCALAR_TYPE_BE(uint32_t, H5::PredType::STD_U32BE);
MAP_H5_SCALAR_TYPE_BE(uint64_t, H5::PredType::STD_U64BE);
MAP_H5_SCALAR_TYPE_BE(int8_t, H5::PredType::STD_I8BE);
MAP_H5_SCALAR_TYPE_BE(int16_t, H5::PredType::STD_I16BE);
MAP_H5_SCALAR_TYPE_BE(int32_t, H5::PredType::STD_I32BE);
MAP_H5_SCALAR_TYPE_BE(int64_t, H5::PredType::STD_I64BE);
MAP_H5_SCALAR_TYPE_BE(float, H5::PredType::IEEE_F32BE);
MAP_H5_SCALAR_TYPE_BE(double, H5::PredType::IEEE_F64BE);
MAP_H5_SCALAR_TYPE_BE(long double, H5::PredType::NATIVE_LDOUBLE);

#undef MAP_H5_SCALAR_TYPE_BE

struct h5_le_traits
{
    typedef le_packer packer_type;
    template<typename T>
    struct get_scalar_data_type: get_h5_scalar_data_type_le<T> {};
};

struct h5_be_traits
{
    typedef be_packer packer_type;
    template<typename T>
    struct get_scalar_data_type: get_h5_scalar_data_type_be<T> {};
};

template<typename T, typename Tfield, typename Tstorage = Tfield>
class field_descriptor
{
public:
    typedef T container_type;
    typedef Tfield field_type;
    typedef Tstorage storage_type;

    typedef boost::variant<
        field_type (container_type::*)() const,
        field_type const& (container_type::*)() const,
        field_type (*)(container_type const&),
        field_type const& (*)(container_type const&),
        field_type (*)(container_type const*),
        field_type const& (*)(container_type const*)> getter_type;

    typedef boost::variant<
        boost::none_t,
        void (container_type::*)(field_type),
        void (container_type::*)(field_type const&),
        field_type& (container_type::*)(),
        void (*)(container_type&, field_type),
        void (*)(container_type&, field_type const&),
        field_type& (*)(container_type&),
        void (*)(container_type*, field_type),
        void (*)(container_type*, field_type const&),
        field_type& (*)(container_type*)> setter_type;

    typedef std::pair<getter_type, setter_type> getter_setter_pair;

    typedef boost::variant<
        field_type container_type::*,
        getter_setter_pair> accessor_type;

public:
    std::string const& field_name() const { return field_name_; }

    accessor_type const& accessor() const { return accessor_; }


    field_descriptor(const char *field_name)
        : field_name_(field_name) {}

    field_descriptor(const char *field_name, field_type container_type::*field)
        : field_name_(field_name), accessor_(field) {}

    field_descriptor(const char *field_name, getter_type const& getter, setter_type const& setter)
        : field_name_(field_name), accessor_(getter_setter_pair(getter, setter)) {}
 
protected:
    std::string field_name_;
    accessor_type accessor_;
};

template<typename Tfield_descriptor_>
struct field_repr
{
    typedef Tfield_descriptor_ descriptor_type;
    typedef typename descriptor_type::container_type container_type;
    typedef typename descriptor_type::field_type field_type;

    field_repr(Tfield_descriptor_ const& descriptor,
               container_type const* container)
            : descriptor(descriptor), container(container) {}

    field_repr(Tfield_descriptor_ const& descriptor,
               void*, field_type const& field_value)
            : descriptor(descriptor), container(0), field_value(field_value) {}

    descriptor_type descriptor;
    container_type const* container;
    boost::optional<field_type const&> field_value;
};

template<typename Tfield_descriptor>
field_repr<Tfield_descriptor> make_field_repr(Tfield_descriptor const& descriptor, typename Tfield_descriptor::container_type const* data)
{
    return field_repr<Tfield_descriptor>(descriptor, data);
}

template<typename Tfield_descriptor>
field_repr<Tfield_descriptor> make_field_repr(Tfield_descriptor const& descriptor, typename Tfield_descriptor::field_type const& field_value)
{
    return field_repr<Tfield_descriptor>(descriptor, 0, field_value);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name), data);
}

template<typename T, typename Tstorage, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield T::*field, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, field), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, boost::none), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield T::*field, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, field), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, boost::none), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (T::*setter)(Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (T::*setter)(Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (T::*setter)(), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T&, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T&, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (*setter)(T&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T*, Tfield), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), void (*setter)(T*, Tfield const&), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (T::*getter)() const, Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (T::*getter)() const, Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const&), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const&), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield (*getter)(T const*), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield, Tstorage>(field_name, getter, setter), data);
}

template<typename Tstorage, typename T, typename Tfield>
field_repr<field_descriptor<T, Tfield, Tstorage> >
field(const char *field_name, Tfield const& (*getter)(T const*), Tfield& (*setter)(T*), T const* data)
{
    return make_field_repr(field_descriptor<T, Tfield>(field_name, getter, setter), data);
}


template<typename Ttraits_>
class field_descriptor_collector
{
public:
    typedef Ttraits_ traits_type;

private:
    struct field
    {
        std::string name;
        std::size_t offset;
        H5::DataType type;
        field* next;

        field(std::string const& name, std::size_t offset, H5::DataType type)
            : name(name), offset(offset), type(type), next(0) {}
    };

public:
    ~field_descriptor_collector()
    {
        H5::CompType retval(offset_);
        for (field* f = fields_, *next = 0; f; f = next)
        {
            retval.insertMember(f->name, f->offset, f->type);
            next = f->next;
            delete f;
        }
        h5type_ = retval;
    }

    template<typename T, typename Tfield, typename Tstorage>
    field_descriptor_collector&
    operator<<(field_repr<field_descriptor<T, Tfield, Tstorage> > const& repr)
    {
        field* f(new field(repr.descriptor.field_name(), offset_,
                typename traits_type::template get_scalar_data_type<Tstorage>()()));
        if (last_)
        {
            last_->next = f;
        }
        else
        {
            fields_ = f;
        }
        last_ = f;
        offset_ += sizeof(Tstorage);
        return *this;
    }

    field_descriptor_collector(H5::CompType& h5type)
        : h5type_(h5type), offset_(0), fields_(0), last_(0) {}
private:
    H5::CompType& h5type_;
    std::size_t offset_;
    field* fields_;
    field* last_;
};

template<typename Ttraits, template<typename> class TTserialize_>
H5::CompType get_h5_type()
{
    H5::CompType retval;
    typedef field_descriptor_collector<Ttraits> collector_type;
    collector_type collector(retval);
    TTserialize_<collector_type>()(collector, 0);
    return retval;
}

template<typename Ttraits, typename T>
H5::DataType get_h5_type()
{
    return typename Ttraits::template get_scalar_data_type<T>()();
}

template<typename Tdest, typename Tsrc>
struct copier
{
    void operator()(Tdest& dest, Tsrc const& src) const
    {
        dest = src;
    }
};

template<typename T, std::size_t N>
struct copier<T[N], T const*>
{
    void operator()(T(&dest)[N], T const* src) const
    {
        *reinterpret_cast<storage_for<T[N]>*>(dest) = *reinterpret_cast<storage_for<T[N]>*>(src);
    }
};

template<std::size_t N>
struct copier<char[N], char const*>
{
    void operator()(char(&dest)[N], char const* src) const
    {
        std::size_t len(std::strlen(src));
        BOOST_ASSERT(len < N - 1);
        std::memcpy(dest, src, len + 1);
    }
};

template<std::size_t N>
struct copier<char[N], std::string>
{
    void operator()(char(&dest)[N], std::string const& src) const
    {
        BOOST_ASSERT(src.size() < N - 1);
        std::memcpy(dest, src.c_str(), src.size() + 1);
    }
};


template<typename Tdest, typename Tsrc>
static void copy_to(Tdest& dest, Tsrc const& src)
{
    copier<Tdest, Tsrc>()(dest, src);
}

template<typename Ttraits>
class field_packer
{
public:
    typedef Ttraits traits_type;

private:
    template<typename Tfield_descriptor>
    struct get_visitor: boost::static_visitor<typename Tfield_descriptor::field_type>
    {
        typedef Tfield_descriptor field_descriptor_type;
        typedef typename field_descriptor_type::container_type container_type;
        typedef typename field_descriptor_type::field_type field_type;
        typedef typename field_descriptor_type::getter_setter_pair getter_setter_pair;
        typedef typename field_descriptor_type::getter_type getter_type;

        struct accessor_visitor: boost::static_visitor<field_type>
        {
            field_type operator()(field_type (container_type::*ptr)() const) const 
            {
                return (container_.*ptr)();
            }

            field_type operator()(field_type const& (container_type::*ptr)() const) const
            {
                return (container_.*ptr)();
            }

            field_type operator()(field_type (*ptr)(container_type const&)) const
            {
                return (*ptr)(container_);
            }

            field_type operator()(field_type const& (*ptr)(container_type const&)) const
            {
                return (*ptr)(container_);
            }

            field_type operator()(field_type (*ptr)(container_type const*)) const
            {
                return (*ptr)(&container_);
            }

            field_type operator()(field_type const& (*ptr)(container_type const*)) const
            {
                return (*ptr)(&container_);
            }

            accessor_visitor(container_type const& container): container_(container) {}

        private:
            container_type const& container_;
        };

        field_type operator()(field_type container_type::*ptr) const
        {
            return container_.*ptr;
        }

        field_type operator()(getter_setter_pair const& pair) const
        {
            return boost::apply_visitor(accessor_visitor(container_), pair.first);
        }

        get_visitor(container_type const& container): container_(container) {}

    private:
        container_type const& container_;
    };

public:
    template<typename T, typename Tfield, typename Tstorage>
    field_packer&
    operator<<(field_repr<field_descriptor<T, Tfield, Tstorage> > const& repr)
    {
        Tstorage value;
        retrieve_field_value(value, repr);
        ptr_ += packer_(ptr_, value);
        return *this;
    }

    unsigned char* next() const
    {
        return ptr_;
    }

    field_packer(unsigned char *buffer): buffer_(buffer), ptr_(buffer_) {}

protected:
    template<typename T, typename Tfield, typename Tstorage>
    static void retrieve_field_value(Tstorage& retval, field_repr<field_descriptor<T, Tfield, Tstorage> > const& repr)
    {
        copy_to(retval,
            repr.field_value ? repr.field_value.get():
                boost::apply_visitor(
                    get_visitor<field_descriptor<T, Tfield, Tstorage> >(*repr.container),
                    repr.descriptor.accessor()));
    }

protected:
    unsigned char *buffer_;
    unsigned char *ptr_;
    typename traits_type::packer_type packer_;
};

template<typename Ttraits, typename T>
unsigned char *pack(unsigned char* buffer, T const& data)
{
    return buffer + typename Ttraits::packer_type()(buffer, data);
}

template<typename Ttraits, template<typename> class TTserialize_, typename T>
unsigned char *pack(unsigned char* buffer, T const& container)
{
    typedef field_packer<Ttraits> _packer_type;
    _packer_type _packer(buffer);
    TTserialize_<_packer_type>()(_packer, &container);
    return _packer.next();
}

#endif /* H5SUPPORT_HPP */
