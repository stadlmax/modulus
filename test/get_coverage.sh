#!/bin/bash

# do the coverage checks
coverage run \
--rcfile='coverage.pytest.rc' \
-m pytest \
--ignore=derivs_test.py \
--ignore=test_meshgraphnet_snmg.py \
--ignore=test_graphcast_snmg.py

coverage run \
--rcfile='coverage.docstring.rc' \
-m pytest \
--doctest-modules ../modulus/

coverage combine --data-file=.coverage
coverage report --omit=*test*

# if you wish to view the report in HTML format uncomment below
# coverage html --omit=*test*

# cleanup
rm .coverage
