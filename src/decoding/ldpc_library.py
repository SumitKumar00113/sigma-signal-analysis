"""Standard LDPC codes: catalogue, download-on-request, local store.

Parity-check matrices of standard codes are not bundled with Sigma.  This
module knows where published matrices for common standards live (the
AFF3CT project's configuration files on GitHub, as ``.alist`` / ``.qc``)
and downloads a named code **only when asked**, into
``~/.sigma/ldpc/``.  Any other ``.alist`` / ``.qc`` file can be used
directly or copied into that folder.

Command line::

    python -m src.decoding.ldpc_library list
    python -m src.decoding.ldpc_library fetch "DVB-S2 r1/2 (64800)"
    python -m src.decoding.ldpc_library fetch --all

Each download is parsed before it is kept, so a bad file is rejected.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.decoding.ldpc import LDPCCode, load_ldpc

LDPC_DIR = Path.home() / ".sigma" / "ldpc"
SOURCE_BASE = "https://raw.githubusercontent.com/aff3ct/configuration_files/develop/dec/LDPC/"


@dataclass(frozen=True)
class CatalogueEntry:
    name: str
    remote: str               # path below SOURCE_BASE
    standard: str

    @property
    def filename(self) -> str:
        return Path(self.remote).name


CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry("DVB-S2 r1/2 (64800)", "DVB-S2/DVB-S2_64800x32400.alist", "ETSI EN 302 307"),
    CatalogueEntry("DVB-S2 r2/3 (64800)", "DVB-S2/DVB-S2_64800x21600.alist", "ETSI EN 302 307"),
    CatalogueEntry("DVB-S2 r3/4 (64800)", "DVB-S2/DVB-S2_64800x16200.alist", "ETSI EN 302 307"),
    CatalogueEntry("DVB-S2 r4/5 (64800)", "DVB-S2/DVB-S2_64800x12960.alist", "ETSI EN 302 307"),
    CatalogueEntry("DVB-S2 r9/10 (64800)", "DVB-S2/DVB-S2_64800x6480.alist", "ETSI EN 302 307"),
    CatalogueEntry("Wi-Fi 802.11n r5/6 (648)", "WIFI_540_648.alist", "IEEE 802.11n"),
    CatalogueEntry("WiMAX r1/2 (576)", "WIMAX_288_576.alist", "IEEE 802.16e"),
    CatalogueEntry("WiMAX r5/6 (576)", "WIMAX_480_576.alist", "IEEE 802.16e"),
    CatalogueEntry("WRAN r3/4 (480)", "WRAN_360_480.alist", "IEEE 802.22"),
    CatalogueEntry("10GBASE-T (2048,1723)", "10GBPS-ETHERNET_1723_2048.alist", "IEEE 802.3an"),
    CatalogueEntry("CCSDS (128,64)", "CCSDS_64_128.alist", "CCSDS 231.1"),
    CatalogueEntry("CCSDS AR4JA r1/2 (8192)", "AR4JA_4096_8192.qc", "CCSDS 131.0"),
    CatalogueEntry("5G NR BG1 Z=384", "5G/NR_1_1_384.qc", "3GPP TS 38.212"),
    CatalogueEntry("5G NR BG2 Z=384", "5G/NR_2_1_384.qc", "3GPP TS 38.212"),
    CatalogueEntry("5G NR BG1 Z=48", "5G/NR_1_1_48.qc", "3GPP TS 38.212"),
    CatalogueEntry("5G NR BG2 Z=48", "5G/NR_2_1_48.qc", "3GPP TS 38.212"),
)


def find_entry(name: str) -> CatalogueEntry:
    key = name.strip().lower()
    for e in CATALOGUE:
        if key in (e.name.lower(), e.filename.lower(), Path(e.filename).stem.lower()):
            return e
    raise KeyError(f"No catalogue code named {name!r}")


def local_path(entry: CatalogueEntry, directory: Path | None = None) -> Path:
    return (directory or LDPC_DIR) / entry.filename


def fetch(entry: CatalogueEntry, directory: Path | None = None, base_url: str = SOURCE_BASE,
          timeout: float = 60.0) -> Path:
    """Download *entry* (if not already present), validate it, return its path."""
    dest = local_path(entry, directory)
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=dest.suffix) as tmp:
        with urllib.request.urlopen(base_url + entry.remote, timeout=timeout) as resp:  # noqa: S310
            shutil.copyfileobj(resp, tmp)
        tmp_path = Path(tmp.name)
    try:
        load_ldpc(tmp_path)               # reject anything that does not parse
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    shutil.move(str(tmp_path), dest)
    return dest


def installed(directory: Path | None = None) -> list[Path]:
    d = directory or LDPC_DIR
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in (".alist", ".qc"))


def display_name(path: Path) -> str:
    for e in CATALOGUE:
        if e.filename == path.name:
            return e.name
    return path.stem


def load_installed(directory: Path | None = None) -> list[LDPCCode]:
    """All installed codes that parse (named after the catalogue when known)."""
    codes = []
    for p in installed(directory):
        try:
            c = load_ldpc(p)
        except Exception:  # noqa: BLE001 – skip unreadable files
            continue
        c.name = display_name(p)
        codes.append(c)
    return codes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show the catalogue and what is installed")
    f = sub.add_parser("fetch", help="download codes from the catalogue")
    f.add_argument("names", nargs="*")
    f.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        have = {p.name for p in installed()}
        for e in CATALOGUE:
            mark = "✓" if e.filename in have else " "
            print(f"[{mark}] {e.name:28s} {e.standard:18s} {e.filename}")
        print(f"\nLocal folder: {LDPC_DIR}")
        return 0
    entries = list(CATALOGUE) if args.all else [find_entry(n) for n in args.names]
    if not entries:
        ap.error("name a code or use --all")
    for e in entries:
        try:
            p = fetch(e)
            print(f"✓ {e.name}: {load_ldpc(p).describe()}  →  {p}")
        except Exception as exc:  # noqa: BLE001
            print(f"✗ {e.name}: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
