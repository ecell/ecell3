Summary: E-Cell is a generic software package for cellular modeling and simulation.
Name: ecell
Version: 3.1.0
Release: 1
URL: http://www.e-cell.org
Source0: %{name}-%{version}.tar.gz
Source1: boost_all.tar.gz
#License: GPL
Group: Applications
Copyright: E-Cell Project
Packager: Takeshi Sakurada
BuildRoot: %{_tmppath}/%{name}-root

%description
E-Cell Simulation Environment (E-Cell SE) is a software package for cellular and biochemical modeling and simulation. E-Cell attempts to provide a framework not only for analyzing metabolism, but also for higher-order cellular phenomena such as gen regulation networks, DNA replication and other occurences in the cell cycle.

%prep
%setup 
%setup -a 1
mv boost_* boost

CFLAGS="$RPM_OPT_FLAGS" ./configure --prefix=%{_prefix}

%build
make


%install
#rm -rf $RPM_BUILD_ROOT
make prefix="$RPM_BUILD_ROOT/usr" install

%clean
rm -rf $RPM_BUILD_ROOT

%files
%defattr(-,root,root)
/usr/bin/*
/usr/lib/*
/usr/share/*


%changelog
* Sun Oct 13 2002 Takeshi Sakurada <sakurada@lupin.e-cell.org>
- Initial build.


