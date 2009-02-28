samplesdir = $(docdir)/samples

SAMPLES = \
	CoupledOscillator \
	Drosophila-cpp \
	Drosophila \
	Heinrich \
	LTD \
	Pendulum \
	SSystem \
	Toy_Hybrid \
	branchG \
	ga \
	heatshock \
	sessionmanager \
	simple \
	tauleap

EXTRA_DIST = $(SAMPLES)

install-data-hook:
	[ ! -e "$(DESTDIR)$(samplesdir)" ] && mkdir "$(DESTDIR)$(samplesdir)"; \
	for dir in $(SAMPLES); do \
		[ ! -e "$(DESTDIR)$(samplesdir)/$$dir" ] && \
			mkdir "$(DESTDIR)$(samplesdir)/$$dir"; \
		for file in "$$dir"/*; do \
			if [ -f "$$file" ]; then \
				$(INSTALL_DATA) "$$file" "$(DESTDIR)$(samplesdir)/$$dir"; \
			elif [ -d "$$file" ]; then \
				[ ! -e "$(DESTDIR)$(samplesdir)/$$file" ] && \
					mkdir "$(DESTDIR)$(samplesdir)/$$file"; \
				for file in "$$file"/*; do \
					$(INSTALL_DATA) "$$file" "$(DESTDIR)$(samplesdir)/$$file"; \
				done; \
			fi; \
		done; \
	done

uninstall-hook:
	$(RM) -r $(DESTDIR)$(samplesdir)

dist-hook:
	for dir in $(SAMPLES); do \
		$(RM) -r "$(distdir)/$$dir/.svn"; \
	done
