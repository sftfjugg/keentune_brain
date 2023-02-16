VERSION 	= 2.0.1
PREFIX     ?= /usr
CONFDIR    ?= /etc
OUTPATH	    = ./bin
TPMPATH 	= $(DESTDIR)/tmp/KEENTUNE
BINDIR      = $(DESTDIR)$(PREFIX)/bin
LOCALBINDIR = $(DESTDIR)$(PREFIX)/local/bin
SYSCONFDIR  = $(DESTDIR)$(CONFDIR)/keentune/conf/
SYSTEMDDIR  = $(DESTDIR)$(PREFIX)/lib/systemd/system

all: target

target:
	pyinstaller --clean --onefile \
		--workpath $(TPMPATH) \
		--distpath $(OUTPATH) \
		--specpath $(TPMPATH) \
		--hidden-import brain.algorithm.tunning.base \
		--hidden-import brain.algorithm.tunning.tpe \
		--hidden-import brain.algorithm.tunning.hord \
		--hidden-import brain.algorithm.tunning.random \
		--hidden-import brain.algorithm.sensitize.sensitize \
		--hidden-import brain.algorithm.sensitize.sensitizer \
		--name keentune-brain \
		brain/brain.py

clean:
	rm -rf $(TPMPATH)
	rm -rf $(OUTPATH)
	rm -rf $(BINDIR)/keentune-brain
	rm -rf $(LOCALBINDIR)/keentune-brain
	rm -rf keentune-brain-$(VERSION).tar.gz

install: 
	@echo "+ Start installing KeenTune-brain"
	mkdir -p $(SYSCONFDIR)
	mkdir -p $(SYSTEMDDIR)
	install -p -D -m 0644 brain/brain.conf $(SYSCONFDIR)
	install -p -D -m 0644 keentune-brain.service $(SYSTEMDDIR)
	mkdir -p $(BINDIR)
	mkdir -p $(LOCALBINDIR)
	cp $(OUTPATH)/* $(BINDIR)
	cp $(OUTPATH)/* $(LOCALBINDIR)
	@echo "+ Make install Done."

startup:
	systemctl daemon-reload
	systemctl restart keentune-brain

tar:
	mkdir -p keentune-brain-$(VERSION)
	cp  --parents $(OUTPATH)/* \
		keentune-brain.service \
		LICENSE \
		Makefile \
		brain/brain.conf \
		keentune-brain-$(VERSION)
	tar -czvf keentune-brain-$(VERSION).tar.gz keentune-brain-$(VERSION)
	rm -rf keentune-brain-$(VERSION)

run: all install startup
rpm: target tar