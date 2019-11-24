#!/usr/bin/env bash
pycodestyle --max-line-length=89 --ignore=E221,E241,E225,E226,E402,E722,W503,W504 see_rnn tests

pytest -s --cov=see_rnn tests/
