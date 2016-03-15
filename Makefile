# Unit-testing, docs, etc.

VIRTUALENV?=virtualenv

TEST?=nosetests
PEP8?=pep8
PROJECT_DIR=.


all: clean pep8


clean:
	find . -name "*.pyc" -exec rm -rf {} \;
	@echo "done"


pep8:
	$(PEP8) --ignore=E501 --exclude='' -r .


.PHONY: all env pip deps pep8 clean bootstrap
