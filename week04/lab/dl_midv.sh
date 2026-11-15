#!/bin/bash
# MIDV-500 원본(smartengines FTP, 2018) 중 5종만 받는다. 책 Manning 링크는 404.
# FTP가 중간에 끊긴다(curl 56) → 이어받기(-C -)로 될 때까지 반복
D=~/datasets/midv500/zips; mkdir -p $D; cd $D
for f in 04_aut_id 09_chn_id 16_deu_passport_new 31_jpn_drvlic 47_usa_bordercrossing; do
  for try in $(seq 1 30); do
    unzip -tq $f.zip >/dev/null 2>&1 && break
    curl -s -C - -o $f.zip ftp://smartengines.com/midv-500/dataset/$f.zip; rc=$?
    [ $rc -ne 0 ] && echo "$f try $try rc=$rc size=$(du -m $f.zip | cut -f1)MB" && sleep 3
  done
  unzip -tq $f.zip >/dev/null 2>&1 && echo "$f ok $(du -h $f.zip | cut -f1)" || { echo "$f FAILED"; exit 1; }
done
cd .. && for f in zips/*.zip; do unzip -qo $f -d raw; done
du -sh raw; ls raw
