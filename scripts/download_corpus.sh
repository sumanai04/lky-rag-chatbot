#!/usr/bin/env bash
# Downloads the raw LKY corpus into data/ and extracts speech text from PDFs.
# Run: bash scripts/download_corpus.sh
set -euo pipefail

mkdir -p data/speeches

echo "[1/3] The Man and His Ideas (book OCR, Internet Archive)"
curl -sL --max-time 300 "https://archive.org/download/lee-kuan-yew/Lee%20Kuan%20Yew_djvu.txt" -o data/the-man-and-his-ideas.txt

echo "[2/3] Wikiquote and Wikipedia (wikitext)"
curl -sL --max-time 60 "https://en.wikiquote.org/w/index.php?title=Lee_Kuan_Yew&action=raw" -o data/wikiquote-lky.wiki
curl -sL --max-time 60 "https://en.wikipedia.org/w/index.php?title=Lee_Kuan_Yew&action=raw" -o data/wikipedia-lky.wiki

echo "[3/3] National Archives of Singapore speech transcripts"
declare -A urls=(
  ["1964-defence-committee-supply"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19641212.pdf"
  ["1965-melbourne-future-of-malaysia"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19650324a.pdf"
  ["1966-india-state-banquet"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19660902.pdf"
  ["1968-student-leadership-seminar"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19680424.pdf"
  ["1968-youth-festival-opening"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19680720.pdf"
  ["1977-national-day-rally"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/National%20Day%20Rally%20Speech%2013%20Aug%201977.pdf"
  ["1981-zhao-ziyang-dinner"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19810811.pdf"
  ["1985-us-congress"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19851009.pdf"
  ["1988-asne-address"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19880414c.pdf"
  ["1990-china-banquet"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19900811.pdf"
  ["1998-thai-defence-college"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/027-1998-01-21_lky.pdf"
  ["1999-sccci-millennium-dinner"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/1999122802/lky19991228c.pdf"
  ["2005-world-ethics-integrity-forum"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/2005042803/2005042803.pdf"
  ["2010-seth-mydans-nyt-interview"]="https://www.nas.gov.sg/archivesonline/data/pdfdoc/20100920006/transcript_of_minister_mentor_lee_kuan_yew.pdf"
)
for name in "${!urls[@]}"; do
  curl -sL --max-time 90 -A "Mozilla/5.0 (X11; Linux x86_64)" "${urls[$name]}" -o "data/speeches/$name.pdf"
  sleep 1
done

if command -v pdftotext >/dev/null 2>&1; then
  for f in data/speeches/*.pdf; do
    pdftotext -layout "$f" "${f%.pdf}.txt"
  done
  echo "Done. Raw corpus is in data/."
else
  echo "Warning: pdftotext not found. Install poppler-utils and re-run to extract speech text."
fi
