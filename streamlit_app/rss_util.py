"""RSS/Atom 피드 HTTP 수집 및 가벼운 파싱 (표준 라이브러리만 사용)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

import httpx

from streamlit_app.config import HTTP_TIMEOUT, USER_AGENT


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _text(elem: ET.Element | None) -> str:
    if elem is None or elem.text is None:
        return ""
    return re.sub(r"\s+", " ", elem.text).strip()


def fetch_feed_items(feed_url: str) -> list[dict[str, Any]]:
    headers = {"User-Agent": USER_AGENT}
    with httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True, headers=headers) as client:
        r = client.get(feed_url)
        r.raise_for_status()
        text = r.text

    root = ET.fromstring(text)
    root_tag = _local_name(root.tag).lower()
    items: list[dict[str, Any]] = []

    if root_tag == "rss" or root_tag == "rdf":
        for elem in root.iter():
            if _local_name(elem.tag).lower() != "item":
                continue
            title = link = desc = pub = ""
            for child in elem:
                ct = _local_name(child.tag).lower()
                if ct == "title":
                    title = _text(child)
                elif ct == "link":
                    link = (child.text or "").strip() or _text(child)
                elif ct == "description":
                    raw = (child.text or ET.tostring(child, encoding="unicode"))
                    desc = re.sub(r"<[^>]+>", " ", raw)
                    desc = re.sub(r"\s+", " ", desc).strip()[:800]
                elif ct == "pubdate":
                    pub = _text(child)
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "link": link or feed_url,
                    "summary": desc,
                    "published": pub,
                    "source_feed": feed_url,
                }
            )
    elif root_tag == "feed":
        for entry in root.iter():
            if _local_name(entry.tag).lower() != "entry":
                continue
            title_el = link_href = None
            summary_el = updated_el = None
            for child in entry:
                cn = _local_name(child.tag).lower()
                if cn == "title":
                    title_el = child
                elif cn == "link" and child.get("href"):
                    link_href = child.get("href")
                elif cn == "summary":
                    summary_el = child
                elif cn == "content":
                    if summary_el is None:
                        summary_el = child
                elif cn == "updated":
                    updated_el = child
            title = _text(title_el)
            if not title:
                continue
            raw_sum = ""
            if summary_el is not None:
                raw_sum = (summary_el.text or "") or ET.tostring(summary_el, encoding="unicode")
            desc = re.sub(r"<[^>]+>", " ", raw_sum)
            desc = re.sub(r"\s+", " ", desc).strip()[:800]
            items.append(
                {
                    "title": title,
                    "link": str(link_href or feed_url),
                    "summary": desc,
                    "published": _text(updated_el),
                    "source_feed": feed_url,
                }
            )

    return items
