#!/usr/bin/env bash
git grep -nE 'AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{32}|ghp_[A-Za-z0-9]{36}' || \
grep -rInE 'AKIA[0-9A-Z]{16}|sk-[a-zA-Z0-9]{32}|ghp_[A-Za-z0-9]{36}' \
  --include='*.py' --include='*.js' --include='*.env*' . 2>/dev/null \
  || echo "no secret matched"
