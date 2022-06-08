%define anolis_release 1

Name:           keentune-brain
Version:        1.1.0
Release:        %{?anolis_release}%{?dist}
Url:            https://gitee.com/anolis/keentune_brain
Summary:        Auto-Tunning algorithm module of KeenTune
Vendor:         Alibaba
License:        MulanPSLv2
Group:          Development/Languages/Python
Source:         %{name}-%{version}.tar.gz

BuildRequires:  python3-devel
BuildRequires:  python3-setuptools
BUildRequires:	systemd

BuildArch:      noarch

Requires:	python3-tornado
Requires(post): systemd
Requires(preun): systemd
Requires(postun): systemd

%description
Auto-Tunning algorithm module of KeenTune

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

%clean
rm -rf $RPM_BUILD_ROOT

%post
%systemd_post keentune-brain.service
if [ -f "%{_prefix}/lib/systemd/system/keentune-brain.service" ]; then
    systemctl enable keentune-brain.service || :
    systemctl start keentune-brain.service || :
fi

%preun
%systemd_preun keentune-brain.service

%postun
%systemd_postun_with_restart keentune-brain.service

%files -f INSTALLED_FILES
%defattr(-,root,root)
%doc README.md
%license LICENSE
%{_prefix}/lib/systemd/system/keentune-brain.service

%changelog
* Thu Mar 03 2022 Runzhe Wang <15501019889@126.com> - 1.1.0-1
- fix bug: update version to 1.1.0 in setup.py script.
- Add support for GP (in iTuned) in sensitizing algorithms
- Add support for lasso in sensitizing algorithms
- refactor tornado module: replace await by threadpool
- lazy load domain in keentune-target
- fix other bugs

* Sat Jan 01 2022 Runzhe Wang <runzhe.wrz@alibaba-inc.com>- 1.0.1
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