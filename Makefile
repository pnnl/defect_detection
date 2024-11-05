SHELL=bash
.SHELLFLAGS=-o pipefail -c
IMAGE=ttpimage/defect_detection
DOCKERFLAGS=--rm -t
ifneq ($(shell whoami), bamboo)
	DOCKERFLAGS+=-i
endif
DOCKER=docker run $(DOCKERFLAGS) -v $${PWD}:/app $(IMAGE)

.PHONY: docs
docs:
	sphinx-build -M html doc/source doc/build
	#sphinx-build -M latexpdf doc/source doc/build


.PHONY: test
test:
	pytest -v --ignore=sandbox/ --cov=./ --cov-branch --cov-report=html --cov-config=.coveragerc test/ | tee doc/source/_static/doc_test.txt


.PHONY: lint
lint:
	pylint defect_detection || true


.PHONY: wheel
wheel:
	python setup.py bdist_wheel


.PHONY: shell
shell: image
	$(DOCKER) bash


.PHONY: ci
ci: image
	$(DOCKER) make lint test wheel docs


.PHONY: image
image:
	docker build -t $(IMAGE) .
