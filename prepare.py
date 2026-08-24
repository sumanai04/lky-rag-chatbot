"""Prepare raw corpus files into clean plain text plus a sources manifest.

Inputs (built by scripts/download_corpus.sh):
    data/the-man-and-his-ideas.txt   book OCR
    data/wikiquote-lky.wiki          wikitext
    data/wikipedia-lky.wiki          wikitext
    data/speeches/*.txt              pdftotext output

Outputs:
    data/clean/*.txt                 cleaned text, one file per source
    data/sources.json                manifest with titles, dates, sources, URLs
"""

import json
import os
import re
from pathlib import Path

CLEAN_DIR = Path("data/clean")

SPEECH_META = {
    "1964-defence-committee-supply": {
        "title": "Committee of Supply Debate on Defence Estimates",
        "date": "1964-12-12",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19641212.pdf",
    },
    "1965-melbourne-future-of-malaysia": {
        "title": "The Future of Malaysia (Institute of International Affairs, Melbourne)",
        "date": "1965-03-24",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19650324a.pdf",
    },
    "1966-india-state-banquet": {
        "title": "State Banquet with Prime Minister Indira Gandhi",
        "date": "1966-09-02",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19660902.pdf",
    },
    "1968-student-leadership-seminar": {
        "title": "Student Leadership in East Asia (Seminar Speech)",
        "date": "1968-04-24",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19680424.pdf",
    },
    "1968-youth-festival-opening": {
        "title": "Opening of the Singapore Youth Festival",
        "date": "1968-07-20",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19680720.pdf",
    },
    "1977-national-day-rally": {
        "title": "National Day Rally Speech (Excerpts)",
        "date": "1977-08-13",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/National%20Day%20Rally%20Speech%2013%20Aug%201977.pdf",
    },
    "1981-zhao-ziyang-dinner": {
        "title": "Dinner in Honour of Premier Zhao Ziyang of China",
        "date": "1981-08-11",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19810811.pdf",
    },
    "1985-us-congress": {
        "title": "Peace and Progress in East Asia (Joint Meeting of the US Congress)",
        "date": "1985-10-09",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19851009.pdf",
    },
    "1988-asne-address": {
        "title": "Address to the American Society of Newspaper Editors",
        "date": "1988-04-14",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19880414c.pdf",
    },
    "1990-china-banquet": {
        "title": "Banquet Hosted by Premier Li Peng of China",
        "date": "1990-08-11",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/lky19900811.pdf",
    },
    "1998-thai-defence-college": {
        "title": "Lecture at the Thai National Defence College",
        "date": "1998-01-21",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/027-1998-01-21_lky.pdf",
    },
    "1999-sccci-millennium-dinner": {
        "title": "SCCCI Millennium Celebration Dinner",
        "date": "1999-12-28",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/1999122802/lky19991228c.pdf",
    },
    "2005-world-ethics-integrity-forum": {
        "title": "World Ethics and Integrity Forum (Kuala Lumpur)",
        "date": "2005-04-28",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/2005042803/2005042803.pdf",
    },
    "2010-seth-mydans-nyt-interview": {
        "title": "Interview with Seth Mydans, New York Times and IHT",
        "date": "2010-09-01",
        "source": "National Archives of Singapore",
        "url": "https://www.nas.gov.sg/archivesonline/data/pdfdoc/20100920006/transcript_of_minister_mentor_lee_kuan_yew.pdf",
    },
}


def clean_wikitext(text):
    """Strip the most common wikitext markup to readable plain text."""
    text = re.sub(r"\[\[File:.*?\]\]", "", text, flags=re.DOTALL)      # images
    text = re.sub(r"\[\[Image:.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)          # templates
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.DOTALL)  # references
    text = re.sub(r"<[^>]+>", "", text)                               # leftover tags
    text = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", text)        # [[target|label]]
    text = re.sub(r"\[\[([^\]]*)\]\]", r"\1", text)                   # [[target]]
    text = re.sub(r"\[(https?://[^\]\s]*)[^\]]*\]", "", text)         # external links
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"^[#*:;]+", "", text, flags=re.MULTILINE)          # list markers
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_book(text):
    """Clean OCR artifacts from the archive.org djvu text."""
    text = re.sub(r"[ \t]{2,}", " ", text)          # double spacing from OCR
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            lines.append("")
            continue
        if re.fullmatch(r"\d{1,4}", s):             # stray page numbers
            continue
        lines.append(s)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_speech(text):
    """Normalize whitespace in pdftotext output."""
    lines = [re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def main():
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    sources = []

    # 1. Speeches
    for name, meta in SPEECH_META.items():
        src = Path(f"data/speeches/{name}.txt")
        if not src.exists():
            continue
        out = CLEAN_DIR / f"{name}.txt"
        out.write_text(clean_speech(src.read_text(encoding="utf-8", errors="ignore")), encoding="utf-8")
        sources.append({"id": name, "file": f"{name}.txt", "kind": "speech", **meta})

    # 2. Wikiquote
    if Path("data/wikiquote-lky.wiki").exists():
        text = clean_wikitext(Path("data/wikiquote-lky.wiki").read_text(encoding="utf-8", errors="ignore"))
        (CLEAN_DIR / "wikiquote-lky.txt").write_text(text, encoding="utf-8")
        sources.append({
            "id": "wikiquote-lky", "file": "wikiquote-lky.txt", "kind": "quotes",
            "title": "Lee Kuan Yew, Quotes by Decade (Wikiquote)", "date": "1950-2015",
            "source": "Wikiquote", "url": "https://en.wikiquote.org/wiki/Lee_Kuan_Yew",
        })

    # 3. Wikipedia
    if Path("data/wikipedia-lky.wiki").exists():
        text = clean_wikitext(Path("data/wikipedia-lky.wiki").read_text(encoding="utf-8", errors="ignore"))
        (CLEAN_DIR / "wikipedia-lky.txt").write_text(text, encoding="utf-8")
        sources.append({
            "id": "wikipedia-lky", "file": "wikipedia-lky.txt", "kind": "biography",
            "title": "Lee Kuan Yew (Wikipedia biographical article)", "date": "2026-08-24",
            "source": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Lee_Kuan_Yew",
        })

    # 4. Book OCR
    if Path("data/the-man-and-his-ideas.txt").exists():
        text = clean_book(Path("data/the-man-and-his-ideas.txt").read_text(encoding="utf-8", errors="ignore"))
        (CLEAN_DIR / "the-man-and-his-ideas.txt").write_text(text, encoding="utf-8")
        sources.append({
            "id": "the-man-and-his-ideas", "file": "the-man-and-his-ideas.txt", "kind": "memoir-interviews",
            "title": "The Man and His Ideas: Selected Speeches and Interviews (Han Fook Kwang, Warren Fernandez, Sumiko Tan)",
            "date": "1998",
            "source": "Internet Archive OCR of the Singapore Press Holdings book",
            "url": "https://archive.org/details/lee-kuan-yew",
        })

    Path("data/sources.json").write_text(json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Prepared {len(sources)} sources into {CLEAN_DIR}/ and data/sources.json")


if __name__ == "__main__":
    main()
