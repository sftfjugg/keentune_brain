%define anolis_release 1

Name:           keentune-brain
Version:        2.0.1
Release:        %{?anolis_release}%{?dist}
Url:            https://gitee.com/anolis/keentune_brain
Summary:        Auto-Tuning algorithm module of KeenTune
Vendor:         Alibaba
License:        MulanPSLv2
Group:          Development/Languages/Python
Source:         %{name}-%{version}.tar.gz

BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
BUildRequires:	systemd

BuildArch:      noarch

Requires:	python3-tornado, python3-numpy
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd

%description
Auto-Tuning algorithm module of KeenTune

%prep
%autosetup -n %{name}-%{version}

%build
%{__python3} setup.py build

%install
%{__python3} setup.py install --single-version-externally-managed -O1 \
			      --prefix=%{_prefix} \
			      --root=%{buildroot} \
 			      --record=INSTALLED_FILES

mkdir -p ${RPM_BUILD_ROOT}/usr/lib/systemd/system/
cp -f ./keentune-brain.service ${RPM_BUILD_ROOT}/usr/lib/systemd/system/
install -D -m 0644 man/keentune-brain.8 ${RPM_BUILD_ROOT}%{_mandir}/man8/keentune-brain.8
install -D -m 0644 man/keentune-brain.conf.5 ${RPM_BUILD_ROOT}%{_mandir}/man5/keentune-brain.conf.5

%clean
rm -rf $RPM_BUILD_ROOT

%post
%systemd_post keentune-brain.service

%preun
%systemd_preun keentune-brain.service

%postun
%systemd_postun keentune-brain.service

%files -f INSTALLED_FILES
%defattr(-,root,root)
%doc README.md
%license LICENSE
%{_prefix}/lib/systemd/system/keentune-brain.service
%{_mandir}/man8/keentune-brain.8*
%{_mandir}/man5/keentune-brain.conf.5*

%changelog
* Fri Jan 13 2023 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 2.0.1-1
- fix randomseed
- fix keentune_brain cannot start without sklearn

* Thu Dec 29 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 2.0.0-0
- fix bug in step adjustment.

* Mon Nov 28 2022 Qinglong Wang <xifu.wql@alibaba-inc.com> - 1.5.0-0
- add algorithm supporting of keenopt

* Mon Oct 31 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.4.0-1
- fix: add requirements of numpy and tornado

* Thu Jul 21 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.3.0-1
- fix: missing of man dir  

* Thu Jun 30 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.3.0-0
- add /avaliable api
- refactor brain.conf

* Mon Jun 27 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.1-3
- fix bug: unnecessary requires e.g. ultraopt, bokeh, requests, paramiko, PyYAML

* Thu Jun 23 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.1-2
- fix bug: no error message if lack of python package in brain

* Mon Jun 20 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.1-1
- update docs

* Mon Apr 04 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.2.0-2
- Wrong version index in python
- Control checking range of settable target for 'profile set'
- add function of all rollback

* Thu Mar 03 2022 Runzhe Wang <15501019889@126.com> - 1.1.0-1
- fix bug: update version to 1.1.0 in setup.py script.
- Add support for GP (in iTuned) in sensitizing algorithms
- Add support for lasso in sensitizing algorithms
- refactor tornado module: replace await by threadpool
- lazy load domain in keentune-target
- fix other bugs

* Tue Feb 01 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com>- 1.0.1
- Verify input arguments of command 'param tune'
- Supporting of multiple target tuning
- Fix bug which cause keentune hanging after command 'param stop'
- Add verification of conflicting commands such as 'param dump', 'param delete' when a tuning job is runing.
- Remove version limitation of tornado
- Refactor sysctl domain to improve stability of parameter setting
- Fix some user experience issues

* Wed Jan 26 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com> - 1.0.0
- remove empty conf dir when uninstall keentune-brain
- fix bug: can not running in alinux2 and centos7
- change modify codeup address to gitee
- add keentune to systemd
- fix: wrong license in setup.py
- use '%license' macro
- update license to MulanPSLv2
- Initial KeenTune-brain
