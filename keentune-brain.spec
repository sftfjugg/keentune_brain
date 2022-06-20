%define anolis_release 1

#
# spec file for package KeenTune-brain
#

Name:           keentune-brain
Version:        1.2.1
Release:        %{?anolis_release}%{?dist}
Url:            https://gitee.com/anolis/keentune_brain
Summary:        AI-Tunning algorithm models for KeenTune
License:        MulanPSLv2
Group:          Development/Languages/Python
Source:         %{name}-%{version}.tar.gz

BuildRoot:      %{_tmppath}/%{name}-%{version}-build
BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
BuildArch:      noarch

Vendor:         Alibaba

%description
AI-Tunning algorithm models for KeenTune

%prep
%setup -q -n %{name}-%{version}

%build
%{__python3} setup.py build

%install
%{__python3} setup.py install --single-version-externally-managed -O1 --prefix=%{_prefix} --root=%{buildroot} --record=INSTALLED_FILES
mkdir -p ${RPM_BUILD_ROOT}/usr/lib/systemd/system/
cp -f ./keentune-brain.service ${RPM_BUILD_ROOT}/usr/lib/systemd/system/

%clean
rm -rf $RPM_BUILD_ROOT

%post
systemctl daemon-reload

%files -f INSTALLED_FILES
%license LICENSE
%{_libdir}/systemd/system/keentune-brain.service

%changelog
* Mon Jun 20 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.1-1
- update docs

* Mon Apr 04 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.0-2
- Wrong version index in python
- Control checking range of settable target for 'profile set'
- add function of all rollback

* Thu Mar 03 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.1.0-2
- fix bug: update version to 1.1.0 in setup.py script.

* Thu Mar 03 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.1.0-1
- Add support for GP (in iTuned) in sensitizing algorithms
- Add support for lasso in sensitizing algorithms
- refactor tornado module: replace await by threadpool
- lazy load domain in keentune-target
- fix other bugs

* Sat Jan 01 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.1-1
- Verify input arguments of command 'param tune'
- Supporting of multiple target tuning
- Fix bug which cause keentune hanging after command 'param stop'
- Add verification of conflicting commands such as 'param dump', 'param delete' when a tuning job is runing.
- Remove version limitation of tornado
- Refactor sysctl domain to improve stability of parameter setting
- Fix some user experience issues

* Sun Dec 12 2021 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0-5
- fix bug: can not running in alinux2 and centos7
- change modify codeup address to gitee

* Wed Dec 01 2021 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0-4
- add keentune to systemd

* Wed Nov 24 2021 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0-3
- fix: wrong license in setup.py

* Wed Nov 10 2021 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0-2
- use '%license' macro
- update license to MulanPSLv2

* Wed Aug 18 2021 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0-1
- Initial KeenTune-brain