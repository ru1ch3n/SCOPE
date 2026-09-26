"""Small, network-free checks for the public landing page."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = []
        self.links = []
        self.images = []
        self.tabs = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for attr in ("href", "src"):
            if attr in attrs:
                self.links.append(attrs[attr])
        if tag == "img":
            self.images.append(attrs)
        if attrs.get("role") == "tab":
            self.tabs.append(attrs)


def parse_page():
    parser = PageParser()
    parser.feed((ROOT / "docs/index.html").read_text(encoding="utf-8"))
    return parser


def test_site_local_links_and_images_exist():
    parser = parse_page()
    assert len(parser.ids) == len(set(parser.ids))
    assert parser.images
    for image in parser.images:
        assert image.get("alt")
    for link in parser.links:
        parts = urlsplit(link)
        if parts.scheme or parts.netloc:
            continue
        if parts.path:
            assert (ROOT / "docs" / unquote(parts.path)).is_file(), link
        if parts.fragment and not parts.path:
            assert parts.fragment in parser.ids, link


def test_site_quickstart_controls_and_scientific_files_are_separate():
    parser = parse_page()
    assert len(parser.tabs) == 3
    assert sum(tab["aria-selected"] == "true" for tab in parser.tabs) == 1
    for tab in parser.tabs:
        assert tab["aria-controls"] in parser.ids
    assert (ROOT / "docs/.nojekyll").is_file()
    assert (ROOT / "docs/PROBES.md").is_file()
    assert (ROOT / "docs/MODELS.md").is_file()
