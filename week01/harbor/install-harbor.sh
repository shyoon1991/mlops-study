#!/bin/bash
# Harbor v2.15.4 online installer — WSL, HTTP, localhost:8085 (2026-10-29)
set -euo pipefail
VER=v2.15.4
cd ~
if [ ! -d ~/harbor ]; then
  curl -fsSL -o /tmp/harbor-online.tgz https://github.com/goharbor/harbor/releases/download/$VER/harbor-online-installer-$VER.tgz
  tar xzf /tmp/harbor-online.tgz -C ~
fi
cd ~/harbor
PWFILE=~/.config/harbor-admin-password
if [ ! -f $PWFILE ]; then
  mkdir -p ~/.config; umask 077
  printf 'H%s9a' "$(openssl rand -hex 12)" > $PWFILE
fi
PW=$(cat $PWFILE)
cp -n harbor.yml.tmpl harbor.yml.orig 2>/dev/null || true
python3 - "$PW" <<'PY'
import sys,re
pw=sys.argv[1]
s=open('harbor.yml.tmpl').read()
s=s.replace('hostname: reg.mydomain.com','hostname: localhost')
s=re.sub(r'(?m)^http:\n  # port for http, default is 80. If https enabled, this port will redirect to https port\n  port: 80',
         'http:\n  # port for http, default is 80. If https enabled, this port will redirect to https port\n  port: 8085',s)
# 로컬 실습이라 HTTPS 블록을 주석 처리
s=re.sub(r'(?m)^https:\n((?:  .*\n|\s*#.*\n)*?)(?=\S)', lambda m: '# https:\n'+''.join('#'+l+'\n' for l in m.group(1).splitlines()), s, count=1)
s=s.replace('harbor_admin_password: Harbor12345','harbor_admin_password: '+pw)
s=s.replace('data_volume: /data','data_volume: /srv/harbor-data')
open('harbor.yml','w').write(s)
PY
chmod 600 harbor.yml
grep -nE '^(hostname|http:|  port|https|# https|data_volume)' harbor.yml
sudo ./install.sh 2>&1 | tail -12
