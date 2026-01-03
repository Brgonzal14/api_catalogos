## ⚙️ Configuración e Imports
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    Query,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import or_
from typing import List, Dict, Tuple, Optional, Any
from contextlib import asynccontextmanager
import os
import io
import time
import zipfile
import xml.etree.ElementTree as ET
import re
import asyncio
from datetime import datetime
import pandas as pd
from fastapi.responses import StreamingResponse
from .db import Base, engine, get_db
from . import models, schemas
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import joinedload
import re
from sqlalchemy import or_, and_, func

# ============================================================
#  Lifespan: crear tablas al iniciar la app (con reintentos)
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Se ejecuta al iniciar la app (antes de aceptar requests) y al cerrarla.
    Aquí hacemos los intentos de conexión a la BD y creación de tablas.
    """
    max_retries = 10
    wait_seconds = 2

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[lifespan] Intentando crear tablas (intento {attempt}/{max_retries})...")
            Base.metadata.create_all(bind=engine)
            print("[lifespan] Tablas creadas correctamente.")
            break
        except OperationalError as e:
            print(f"[lifespan] BD no disponible todavía: {e}")
            if attempt == max_retries:
                print("[lifespan] No se pudo conectar a la BD después de varios intentos. Abortando.")
                raise
            await asyncio.sleep(wait_seconds)

    # Aquí la app ya está lista
    yield

    # Si quisieras lógica al apagar la app, iría después del yield.
    print("[lifespan] Cerrando aplicación.")


## 🚀 Inicialización de la App
app = FastAPI(title="API Catálogos Aeronáuticos", lifespan=lifespan)

# =========================
# Frontend estático
# =========================
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# --- CORS para permitir el front independiente 
app.add_middleware(
     CORSMiddleware,
     allow_origins=["*"],       # para demo lo dejamos abierto
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
 ) 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/", include_in_schema=False)
def read_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))



# ============================================================
#  Helpers de BD
# ============================================================
def get_or_create_supplier(db: Session, name: str) -> models.Supplier:
    """Busca un proveedor por nombre o lo crea si no existe."""
    supplier = db.query(models.Supplier).filter(models.Supplier.name == name).first()
    if supplier:
        return supplier
    supplier = models.Supplier(name=name)
    db.add(supplier)
    db.commit()
    db.refresh(supplier)
    return supplier

def sql_normalize(col, db: Session):
    """
    Normaliza en SQL quitando separadores.
    - Postgres: regexp_replace
    - SQLite: replace encadenado (sin regex)
    """
    dialect = db.bind.dialect.name
    base = func.upper(func.coalesce(col, ""))

    if dialect == "postgresql":
        return func.regexp_replace(base, r"[^A-Z0-9]", "", "g")

    if dialect == "sqlite":
        x = base
        for ch in ["-", " ", ".", "/", "_", "(", ")", "[", "]"]:
            x = func.replace(x, ch, "")
        return x

    return base


def create_catalog(db: Session, supplier: models.Supplier, year: int, filename: str) -> models.Catalog:
    """Crea una nueva entrada de catálogo."""
    catalog = models.Catalog(
        supplier_id=supplier.id,
        year=year,
        original_filename=filename,
    )
    db.add(catalog)
    db.commit()
    db.refresh(catalog)
    return catalog


def normalize_part_number(code: str) -> Tuple[str, str]:
    """Normaliza el código de parte y extrae la raíz."""
    if code is None:
        return "", ""
    code = code.strip()
    root = code.split("-")[0]
    return code, root

import re

_norm_re = re.compile(r"[^A-Za-z0-9]+")

def normalize_pn(s: str) -> str:
    """Deja el código en MAYÚSCULAS y sin separadores (solo A-Z 0-9)."""
    if not s:
        return ""
    return _norm_re.sub("", str(s).upper().strip())



def extract_alias_codes(raw: Optional[str]) -> List[str]:
    """
    Extrae códigos equivalentes desde una celda.
    Soporta:
      - separados por coma / ; / salto de línea
      - paréntesis: "CODE1 (CODE2)" -> ["CODE1", "CODE2"]
    """
    if raw is None:
        return []

    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return []

    # separar por coma / ; / saltos
    chunks = [c.strip() for c in re.split(r"[,\n;]+", s) if c and c.strip()]
    out: List[str] = []

    for c in chunks:
        # CODE1 (CODE2)
        m = re.match(r"^(.*?)\((.*?)\)$", c)
        if m:
            a = m.group(1).strip()
            b = m.group(2).strip()
            if a:
                out.append(a)
            if b:
                out.append(b)
        else:
            out.append(c)

    # dedupe preservando orden
    seen = set()
    deduped = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        deduped.append(x)

    return deduped

def wildcard_x_match(pattern: str, value: str) -> bool:
    """
    Igual que antes, pero comparando en formato normalizado:
    - Elimina guiones/espacios/etc en ambos
    - 'X' significa cualquier alfanumérico
    """
    p = normalize_pn(pattern)
    v = normalize_pn(value)

    if not p or not v or len(p) != len(v):
        return False

    for pc, vc in zip(p, v):
        if pc == "X":
            if not vc.isalnum():
                return False
            continue
        if pc != vc:
            return False
    return True

def format_prices_for_export(part) -> str:
    """
    Devuelve un string con todos los precios:
    - Si hay price_tiers: los lista ordenados (min-max o >=min)
    - Si no hay tiers pero hay base_price: usa min_qty_default
    """
    tiers = list(getattr(part, "price_tiers", []) or [])

    # Si hay tiers, formateamos todos
    if tiers:
        # ordenar por min_qty (None al final)
        tiers.sort(key=lambda x: (x.min_qty is None, x.min_qty or 0))
        chunks = []
        for t in tiers:
            cur = t.currency or part.currency or ""
            if t.max_qty is not None:
                label = f"{t.min_qty}-{t.max_qty}"
            else:
                label = f">={t.min_qty}"
            chunks.append(f"{label}: {t.unit_price} {cur}".strip())
        return " | ".join(chunks)

    # Si no hay tiers, usar base_price
    if part.base_price is not None:
        cur = part.currency or ""
        minq = part.min_qty_default or 1
        return f">={minq}: {part.base_price} {cur}".strip()

    return ""



# ============================================================
#  Helpers de Lectura de Archivos
# ============================================================
def read_xlsx_fallback(xlsx_bytes: bytes) -> pd.DataFrame:
    """
    Lector de respaldo para archivos .xlsx que openpyxl no puede abrir
    (por XML inválido).
    Lee directamente xl/worksheets/sheet1.xml y xl/sharedStrings.xml.
    """
    z = zipfile.ZipFile(io.BytesIO(xlsx_bytes))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    # ----- sharedStrings (texto compartido) -----
    shared_strings: List[str] = []
    try:
        shared_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in shared_root.findall("x:si", ns):
            t_el = si.find("x:t", ns)
            if t_el is not None:
                # texto simple
                shared_strings.append(t_el.text or "")
            else:
                # texto compuesto por varios <r><t>
                text = ""
                for r in si.findall("x:r", ns):
                    t = r.find("x:t", ns)
                    if t is not None and t.text:
                        text += t.text
                shared_strings.append(text)
    except KeyError:
        # libro sin sharedStrings.xml
        shared_strings = []

    # ----- sheet1.xml -----
    sheet_xml = z.read("xl/worksheets/sheet1.xml")
    root = ET.fromstring(sheet_xml)

    rows_data: List[List[Optional[str]]] = []

    for row in root.findall("x:sheetData/x:row", ns):
        row_list: List[Optional[str]] = []
        for c in row.findall("x:c", ns):
            t = c.get("t")  # tipo de celda
            v_el = c.find("x:v", ns)
            if v_el is None:
                val = None
            else:
                v = v_el.text
                if t == "s" and v is not None:
                    # índice en sharedStrings
                    idx = int(v)
                    val = shared_strings[idx] if 0 <= idx < len(shared_strings) else None
                else:
                    val = v
            row_list.append(val)
        rows_data.append(row_list)

    if not rows_data:
        return pd.DataFrame()

    # Igualamos el largo de todas las filas
    max_len = max(len(r) for r in rows_data)
    norm_rows = [r + [None] * (max_len - len(r)) for r in rows_data]

    return pd.DataFrame(norm_rows)


def read_pdf_tables(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Lee todas las tablas de un PDF (caso STUKERJURGEN Hansair)
    y las concatena en un solo DataFrame bruto.
    """
    try:
        import pdfplumber
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail=(
                "El soporte para PDF requiere la librería 'pdfplumber'. "
                "Instálala en el entorno del backend (pip install pdfplumber)."
            ),
        )

    all_rows: List[List[Optional[str]]] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                for row in table:
                    all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    max_len = max(len(r) for r in all_rows)
    norm_rows = [r + [None] * (max_len - len(r)) for r in all_rows]
    return pd.DataFrame(norm_rows)

# ============================================================
#  Helpers PDF (vendor-specific): IPECO / DIEHL
#  - Algunos PDFs NO exponen tablas a pdfplumber, así que los
#    parseamos desde texto usando pypdf.
# ============================================================

def _iter_text_pages_pypdf(pdf_bytes: bytes, max_pages: Optional[int] = None):
    """
    Itera el texto extraído por página.

    Preferimos **PyPDF2** (está en tus requirements). Si no está disponible,
    intentamos con **pypdf** como fallback.
    """
    PdfReader = None
    err_1 = None
    try:
        from PyPDF2 import PdfReader as _PdfReader  # type: ignore
        PdfReader = _PdfReader
    except Exception as e1:
        err_1 = e1
        try:
            from pypdf import PdfReader as _PdfReader  # type: ignore
            PdfReader = _PdfReader
        except Exception as e2:
            raise HTTPException(
                status_code=500,
                detail=(
                    "El soporte PDF (texto) requiere 'PyPDF2' (recomendado) o 'pypdf'. "
                    f"Detalle PyPDF2: {err_1} | Detalle pypdf: {e2}"
                ),
            )

    reader = PdfReader(io.BytesIO(pdf_bytes))

    # algunos PDFs pueden venir cifrados
    try:
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")  # contraseña vacía
            except Exception:
                pass
    except Exception:
        pass

    n = len(reader.pages)
    if max_pages is not None:
        n = min(n, max_pages)

    for i in range(n):
        yield (reader.pages[i].extract_text() or "")


def _first_page_text_pypdf(pdf_bytes: bytes) -> str:
    for t in _iter_text_pages_pypdf(pdf_bytes, max_pages=1):
        return t or ""
    return ""


def _looks_like_ipeco(first_text: str) -> bool:
    t = (first_text or "").lower()
    # El PDF suele decir "IPECO ... Dollar Price List"
    return ("ipeco" in t) or ("dollar price list" in t and "material" in t)


def _looks_like_diehl(first_text: str) -> bool:
    t = (first_text or "").lower()
    # Suele decir "DIEHL Aviation Hamburg GmbH" / "Broker Net Price List"
    return ("diehl" in t) or ("broker net price list" in t)


def parse_pdf_ipeco(pdf_bytes: bytes) -> pd.DataFrame:
    """
    IPECO 2026 Dollar Price List (PDF):
      Material | Description | Price (USD) | Per | UoM | Of | Lead Time (days) | Export Licence
    Nota: se extrae desde texto (pypdf), no desde tablas.
    """
    rows: List[Dict[str, Any]] = []

    for page_text in _iter_text_pages_pypdf(pdf_bytes):
        if not page_text:
            continue

        for line in page_text.splitlines():
            s = (line or "").strip()
            if not s:
                continue

            low = s.lower()
            # saltar cabeceras y títulos
            if low.startswith("material") and ("price" in low or "uom" in low):
                continue
            if "ipeco" in low and "price list" in low:
                continue

            tokens = s.split()
            if len(tokens) < 7:
                continue

            part_no = tokens[0]

            # Encuentra los 2 últimos tokens NUMÉRICOS: Of y LeadTime
            num_idxs = [i for i, tok in enumerate(tokens) if tok.isdigit()]
            if len(num_idxs) < 2:
                continue

            lead_idx = num_idxs[-1]
            of_idx = num_idxs[-2]

            # Estructura esperada al final:
            # ... <price> <per> <uom> <of> <lead> [export_licence...]
            uom_idx = of_idx - 1
            per_idx = uom_idx - 1
            price_idx = per_idx - 1

            if price_idx < 2 or lead_idx <= of_idx:
                continue

            price_raw = tokens[price_idx]
            price_val = parse_price_value(price_raw)
            if price_val is None:
                continue

            description = " ".join(tokens[1:price_idx]).strip()
            per_qty = tokens[per_idx]
            uom = tokens[uom_idx]
            of_qty = tokens[of_idx]
            lead_time = tokens[lead_idx]
            export_lic = " ".join(tokens[lead_idx + 1:]).strip() or None

            rows.append(
                {
                    "Material": part_no,
                    "Description": description,
                    "Price (USD)": price_raw,
                    "Currency": "USD",
                    "Per": per_qty,
                    "UoM": uom,
                    "Of": of_qty,
                    "Lead Time (days)": lead_time,
                    "Export Licence": export_lic,
                }
            )

    return pd.DataFrame(rows)


def parse_pdf_diehl(pdf_bytes: bytes) -> pd.DataFrame:
    """
    DIEHL Broker Price Catalogue (PDF):
      PNR | Description | Effect Date (YYYYMMDD) | UNT | Price 2026 | CUR | MOQ | SPQ | LTM
    Nota: el formato viene "en línea", por eso se extrae desde texto (pypdf).
    """
    rows: List[Dict[str, Any]] = []
    date_re = re.compile(r"^\d{8}$")

    for page_text in _iter_text_pages_pypdf(pdf_bytes):
        if not page_text:
            continue

        for line in page_text.splitlines():
            s = (line or "").strip()
            if not s:
                continue

            low = s.lower()
            # saltar cabeceras
            if low.startswith("pnr ") or low.startswith("pnr\t"):
                continue
            if "broker net price list" in low or "diehl aviation" in low:
                # puede ser cabecera de página
                continue

            tokens = s.split()
            if len(tokens) < 6:
                continue

            pnr = tokens[0]

            # encontrar índice de fecha efecto
            date_idx = None
            for i, tok in enumerate(tokens):
                if date_re.match(tok):
                    date_idx = i
                    break
            if date_idx is None or date_idx < 2:
                continue
            if date_idx + 3 >= len(tokens):
                continue

            desc = " ".join(tokens[1:date_idx]).strip()
            eff_date = tokens[date_idx]
            unt = tokens[date_idx + 1]
            price_raw = tokens[date_idx + 2]
            cur = tokens[date_idx + 3]

            # resto: MOQ [SPQ LTM] o "on request"
            rest = tokens[date_idx + 4:]
            moq = rest[0] if len(rest) >= 1 else None
            spq = None
            ltm = None

            if len(rest) >= 3 and rest[1].isdigit() and rest[2].isdigit():
                spq = rest[1]
                ltm = rest[2]
            else:
                # Ej: "1 on request"
                tail = " ".join(rest[1:]).strip() if len(rest) > 1 else ""
                if tail:
                    ltm = tail

            # validar precio
            if parse_price_value(price_raw) is None:
                continue

            rows.append(
                {
                    "PNR": pnr,
                    "Description": desc,
                    "Effect Date": eff_date,
                    "UNT": unt,
                    "Price 2026": price_raw,
                    "CUR": cur,
                    "MOQ": moq,
                    "SPQ": spq,
                    "LTM": ltm,
                }
            )

    return pd.DataFrame(rows)



# ============================================================
#  Helpers PDF (vendor-specific): AES / MATZEN & TIMM / STABILUS / STUKERJURGEN SAC / VINCORION
#  Nota: algunos de estos PDFs no exponen tablas a pdfplumber, por lo que se parsean por texto.
# ============================================================

def _looks_like_aes(text: str) -> bool:
    t = (text or "").lower()
    return ("distributor" in t and "price list" in t and "part number" in t)

def _looks_like_matzen(text: str) -> bool:
    t = (text or "").lower()
    return ("matzen" in t and "timm" in t) or ("matzen & timm" in t)

def _looks_like_stabilus(text: str) -> bool:
    t = (text or "").lower()
    # Alemán: "Preisliste", "Sonderpreisliste"
    return ("stabilus" in t) or ("sonderpreisliste" in t) or ("preisliste" in t and "art.-nr" in t)

def _looks_like_stuker_sac(text: str) -> bool:
    t = (text or "").lower()
    return ("sac products" in t) or ("bauteil-nr" in t) or ("bezeichnung" in t and "länge" in t)

def _looks_like_vincorion(text: str) -> bool:
    t = (text or "").lower()
    return ("vincorion" in t) or ("annual price catalogue" in t and "spare parts" in t)

def parse_pdf_aes(pdf_bytes: bytes) -> pd.DataFrame:
    """AES Distributor price list (PDF con tablas)."""
    try:
        import pdfplumber
        tables_rows = []
        header = None
        for page_idx in range(0, 10):  # suficiente para captar la lista
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                if page_idx >= len(pdf.pages):
                    break
                page = pdf.pages[page_idx]
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        # normaliza celdas
                        row = [c.strip() if isinstance(c, str) else c for c in row]
                        # detecta header real
                        if row and any(isinstance(c, str) and c.strip().lower() == "part number" for c in row):
                            header = row
                            continue
                        if header is None:
                            continue
                        # filas de datos: primera celda debe parecer P/N (no vacía)
                        if not row or row[0] is None:
                            continue
                        first = str(row[0]).strip()
                        if not first or first.lower().startswith("distributor"):
                            continue
                        tables_rows.append(row)
        if not header or not tables_rows:
            # fallback genérico
            return read_pdf_tables(pdf_bytes)

        # Ajustar longitudes
        max_len = max(len(r) for r in ([header] + tables_rows))
        def _pad(r): return r + [None] * (max_len - len(r))
        header = _pad(header)
        tables_rows = [_pad(r) for r in tables_rows]

        cols = []
        for c in header:
            c = (str(c).replace("\n", " ").strip() if c is not None else "")
            cols.append(c if c else f"col_{len(cols)}")

        df = pd.DataFrame(tables_rows, columns=cols)

        # estandarizar nombres clave (sin perder info: quedará como atributos también)
        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if lc == "part number":
                rename[c] = "part_number"
            elif "description" in lc:
                rename[c] = "description"
            elif "price" in lc:
                rename[c] = "price"
            elif lc.strip() == "moq":
                rename[c] = "min_qty"
            elif "certificate" in lc:
                rename[c] = "certificate"
            elif "length in inch" in lc:
                rename[c] = "length_inch"
            elif "length in cm" in lc:
                rename[c] = "length_cm"
        return df.rename(columns=rename)

    except Exception:
        # fallback genérico
        return read_pdf_tables(pdf_bytes)

def parse_pdf_matzen(pdf_bytes: bytes) -> pd.DataFrame:
    """MATZEN & TIMM (AIRBUS): Airbus P/N principal y Supplier P/N como alias."""
    try:
        import pdfplumber
        rows = []
        header = None
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    for row in table:
                        row = [c.strip() if isinstance(c, str) else c for c in row]
                        # header contiene "Customer P/N" y "Description"
                        if row and any(isinstance(c, str) and "customer p/n" in c.lower() for c in row) and any(isinstance(c, str) and "description" in c.lower() for c in row):
                            header = row
                            continue
                        if header is None:
                            continue
                        if not row or row[0] is None:
                            continue
                        # ignora paginación / títulos
                        if isinstance(row[0], str) and row[0].lower().startswith("spare parts"):
                            continue
                        rows.append(row)

        if not header or not rows:
            return read_pdf_tables(pdf_bytes)

        max_len = max(len(r) for r in ([header] + rows))
        def _pad(r): return r + [None] * (max_len - len(r))
        header = _pad(header)
        rows = [_pad(r) for r in rows]

        cols = []
        for c in header:
            c = (str(c).replace("\n", " ").strip() if c is not None else "")
            cols.append(c if c else f"col_{len(cols)}")

        df = pd.DataFrame(rows, columns=cols)

        # Renames: Airbus PN principal
        rename = {}
        for c in df.columns:
            lc = str(c).lower()
            if "customer p/n" in lc:
                rename[c] = "part_number"  # AIRBUS P/N
            elif "m&t p/n" in lc or "m&t" in lc and "p/n" in lc:
                rename[c] = "supplier_part_no"  # alias
            elif lc.strip() == "customer":
                rename[c] = "customer"
            elif "description" in lc:
                rename[c] = "description"
            elif "ata" in lc:
                rename[c] = "ata_chapter"
            elif "shelf" in lc:
                rename[c] = "shelf_life"
            elif "price" in lc:
                rename[c] = "price"
            elif "pack" in lc and "size" in lc:
                rename[c] = "pack_qty"
            elif "qty" in lc and "pack" in lc:
                rename[c] = "pack_qty"
        return df.rename(columns=rename)

    except Exception:
        return read_pdf_tables(pdf_bytes)

def parse_pdf_stabilus(pdf_bytes: bytes) -> pd.DataFrame:
    """STABILUS (Preisliste): filas por tramo (ab Menge / VK)."""
    try:
        import pdfplumber
        items = {}  # pn -> list[(min_qty, price)]
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables() or []
                for table in tables:
                    # detectar header
                    header_idx = None
                    for i, row in enumerate(table[:5]):
                        if not row:
                            continue
                        low = " ".join([str(c).lower() for c in row if c is not None])
                        if "art" in low and "nr" in low and ("vk" in low or "menge" in low):
                            header_idx = i
                            break
                    if header_idx is None:
                        continue
                    header = table[header_idx]
                    # column indexes
                    col_map = {}
                    for j, c in enumerate(header):
                        lc = (str(c).lower() if c is not None else "")
                        if "art" in lc and "nr" in lc:
                            col_map["pn"] = j
                        elif "ab" in lc and "menge" in lc:
                            col_map["min"] = j
                        elif "vk" in lc:
                            col_map["price"] = j
                    if "pn" not in col_map or "min" not in col_map or "price" not in col_map:
                        continue

                    for row in table[header_idx+1:]:
                        if not row or len(row) <= max(col_map.values()):
                            continue
                        pn = row[col_map["pn"]]
                        min_q = row[col_map["min"]]
                        pr = row[col_map["price"]]
                        if pn is None:
                            continue
                        pn = str(pn).strip()
                        if not pn or pn.lower().startswith("art"):
                            continue
                        # limpiar min qty
                        min_qty = None
                        if min_q is not None:
                            m = re.search(r"\d+", str(min_q))
                            if m:
                                min_qty = int(m.group(0))
                        price = parse_price_value(pr)
                        if min_qty is None and price is None:
                            continue
                        items.setdefault(pn, [])
                        if min_qty is None:
                            min_qty = 1
                        if price is not None:
                            items[pn].append((min_qty, price))

        rows = []
        for pn, tiers in items.items():
            tiers = sorted({(mq, pr) for mq, pr in tiers}, key=lambda x: x[0])
            for idx, (mq, pr) in enumerate(tiers):
                next_mq = tiers[idx+1][0] if idx+1 < len(tiers) else None
                max_qty = (next_mq - 1) if next_mq is not None else None
                rows.append({
                    "part_number": pn,
                    "min_qty": mq,
                    "max_qty": max_qty,
                    "price": pr,
                    "currency": "EUR",
                })
        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame()

def _parse_qty_token(token: str) -> Tuple[Optional[int], Optional[str]]:
    """Token tipo '100St.' / '100,0' -> (100, 'St.')"""
    if token is None:
        return (None, None)
    s = str(token).strip()
    if not s:
        return (None, None)
    m = re.match(r"^(?P<qty>\d+(?:[\.,]\d+)?)\s*(?P<u>[A-Za-z\.]+)?$", s)
    if not m:
        return (None, None)
    qty_raw = m.group("qty")
    u = m.group("u")
    # qty
    try:
        q = float(qty_raw.replace(".", "").replace(",", "."))
        qty_i = int(round(q))
        return (qty_i, u)
    except Exception:
        return (None, u)

def parse_pdf_stuker_sac(pdf_bytes: bytes) -> pd.DataFrame:
    """STUKERJURGEN SAC Products: parseo por texto (alemán)"""
    rows = []
    try:
        for page_text in _iter_text_pages_pypdf(pdf_bytes):
            if not page_text:
                continue
            for line in page_text.splitlines():
                raw = (line or "").strip()
                if not raw:
                    continue
                low = raw.lower()
                if low.startswith("p/n") or "bauteil-nr" in low or "variante" in low or "bezeichnung" in low:
                    continue
                if not re.match(r"^\d{4,}\s+", raw):
                    continue

                # normaliza typos comunes
                raw = re.sub(r"\bmon\s+request\b", "on request", raw, flags=re.IGNORECASE)

                # extraer precio al final (con posible unidad pegada: 'm36,10', o qty+unit+price: '100St.22,75')
                m = re.search(r"(?P<pre>[A-Za-z\.]*)(?P<price>\d+(?:\.\d{3})*(?:,\d+)?)\s*$", raw)
                if not m:
                    # on request
                    if "on request" in low:
                        # intentamos igualmente sacar los campos básicos
                        tokens = raw.split()
                        if len(tokens) < 6:
                            continue
                        pn = tokens[0]
                        supplier = tokens[1]
                        variant = tokens[2]
                        # asume length y qty al final
                        length = tokens[-2]
                        qty_token = tokens[-1]
                        qty_i, qty_u = _parse_qty_token(qty_token)
                        desc = " ".join(tokens[3:-2]).strip()
                        rows.append({
                            "part_number": pn,
                            "supplier_part_no": supplier,
                            "variant": variant,
                            "description": desc,
                            "length_mm": length,
                            "min_qty": qty_i,
                            "uom": qty_u,
                            "pricing_note": "on request",
                            "currency": "EUR",
                        })
                    continue

                pre = (m.group("pre") or "").strip()
                price_raw = (m.group("price") or "").strip()
                line_wo_price = raw[:m.start()].rstrip()

                unit_from_price = pre if pre else None

                tokens = line_wo_price.split()
                if len(tokens) < 6:
                    continue

                pn = tokens[0]
                supplier = tokens[1]
                variant = tokens[2]

                qty_token = tokens[-1]
                length = tokens[-2]

                qty_i, qty_u = _parse_qty_token(qty_token)
                uom = unit_from_price or qty_u

                desc = " ".join(tokens[3:-2]).strip()

                price = parse_price_value(price_raw)
                if price is None:
                    continue

                rows.append({
                    "part_number": pn,
                    "supplier_part_no": supplier,
                    "variant": variant,
                    "description": desc,
                    "length_mm": length,
                    "min_qty": qty_i,
                    "uom": uom,
                    "price": price,
                    "currency": "EUR",
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(rows)

def parse_pdf_vincorion(pdf_bytes: bytes) -> pd.DataFrame:
    """VINCORION Airbus spares: parseo por texto (ATA | Partnumber | Designation | Lead Time | Price US-$)."""
    rows = []
    try:
        header_seen = False
        for page_text in _iter_text_pages_pypdf(pdf_bytes):
            if not page_text:
                continue
            for line in page_text.splitlines():
                raw = (line or "").strip()
                if not raw:
                    continue
                low = raw.lower()

                if "ata partnumber designation of item lead time price" in low:
                    header_seen = True
                    continue
                if not header_seen:
                    continue

                # cortar cuando viene otro bloque
                if low.startswith("annual price catalogue"):
                    continue
                if "replacements" in low and "ata" not in low:
                    continue
                if low.startswith("control units") or low.startswith("hydraulic") or low.startswith("electrical"):
                    # títulos de sección
                    continue

                # líneas válidas empiezan con ATA (2 dígitos)
                m = re.match(r"^(?P<ata>\d{2})\s+(?P<pn>[A-Za-z0-9\-\/]+)\s+(?P<rest>.+)$", raw)
                if not m:
                    continue
                ata = m.group("ata")
                pn = m.group("pn").strip()
                rest = m.group("rest").strip()

                # precio: último token numérico
                toks = rest.split()
                if not toks:
                    continue
                price_token = toks[-1]
                price = parse_price_value(price_token)
                if price is None:
                    continue
                rest_wo_price = " ".join(toks[:-1]).strip()

                lead_time = None
                # buscar "on request"
                if re.search(r"\bon\s+request\b", rest_wo_price, flags=re.IGNORECASE):
                    lead_time = "on request"
                    desc = re.sub(r"\bon\s+request\b", "", rest_wo_price, flags=re.IGNORECASE).strip()
                else:
                    mlt = re.search(r"(\d+\s*(?:day|days|week|weeks|month|months))", rest_wo_price, flags=re.IGNORECASE)
                    if mlt:
                        lead_time = mlt.group(1)
                        desc = rest_wo_price[:mlt.start()].strip()
                    else:
                        desc = rest_wo_price

                rows.append({
                    "ata": ata,
                    "part_number": pn,
                    "description": desc,
                    "lead_time": lead_time,
                    "price": price,
                    "currency": "USD",
                })

        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(rows)




def read_pdf_vendor_dfs(pdf_bytes: bytes) -> List[pd.DataFrame]:
    """
    Decide el parser PDF según el contenido del PDF.
    - IPECO / DIEHL: parseo por texto (pypdf)
    - fallback: pdfplumber tables (read_pdf_tables)
    """
    first = _first_page_text_pypdf(pdf_bytes)

    # Algunos PDFs (p.ej. TDI) extraen mejor texto con pdfplumber.
    pl_text = _first_page_text_pdfplumber(pdf_bytes)
    if pl_text and (len(pl_text.strip()) > len((first or "").strip())):
        first = pl_text

    try:
        # TDI (Meridian / Spectrum) - PDF por texto
        if _looks_like_tdi(first):
            df = parse_pdf_tdi(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_ipeco(first):
            df = parse_pdf_ipeco(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_diehl(first):
            df = parse_pdf_diehl(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        # Nuevos proveedores 2026 (PDF):
        if _looks_like_aes(first):
            df = parse_pdf_aes(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_matzen(first):
            df = parse_pdf_matzen(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_stabilus(first):
            df = parse_pdf_stabilus(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_stuker_sac(first):
            df = parse_pdf_stuker_sac(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

        if _looks_like_vincorion(first):
            df = parse_pdf_vincorion(pdf_bytes)
            if df is not None and not df.empty:
                return [df]

    except Exception as e:
        # Si el parser específico falla, intentamos con tablas
        print("[read_pdf_vendor_dfs] Parser específico falló, intentando tablas:", e)

    # fallback: tablas
    return [read_pdf_tables(pdf_bytes)]



# ============================================================
#  Helpers de normalización extra (columnas duplicadas / tiers por encabezado)
# ============================================================
def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Asegura nombres de columnas únicos (pandas permite duplicados y rompe .get / ffill)."""
    cols = []
    counts: Dict[str, int] = {}
    for c in list(df.columns):
        base = str(c)
        n = counts.get(base, 0) + 1
        counts[base] = n
        cols.append(base if n == 1 else f"{base}__{n}")
    df.columns = cols
    return df

_tier_range_re = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_tier_num_re = re.compile(r"^\s*(\d+)\s*$")

def parse_tier_header(header: Any) -> Tuple[Optional[int], Optional[int]]:
    """Interpreta encabezados tipo '5-9', '10 - 24' o '500' como tramos de precios."""
    if header is None:
        return (None, None)
    if isinstance(header, (int, float)) and not pd.isna(header):
        # 500.0 -> 500
        if float(header).is_integer():
            header = str(int(header))
        else:
            header = str(header)
    s = str(header).strip()
    if not s:
        return (None, None)
    # 5-9
    m = _tier_range_re.match(s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    # 500  -> asumimos >=500 (max None)
    m = _tier_num_re.match(s)
    if m:
        return (int(m.group(1)), None)
    return (None, None)

_moq_re = re.compile(r"\bmoq\b[^0-9]*(\d+)", flags=re.IGNORECASE)

def parse_moq_from_text(raw: Any) -> Optional[int]:
    """Extrae MOQ desde texto tipo 'MOQ 100 EA' / 'MOQ: 10'"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return None
    m = _moq_re.search(s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


# ============================================================
#  Helpers PDF (vendor-specific): TDI (PDF por texto, no tablas)
# ============================================================
def _first_page_text_pdfplumber(pdf_bytes: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return ""
            return (pdf.pages[0].extract_text() or "")
    except Exception:
        return ""

def _looks_like_tdi(text: str) -> bool:
    t = (text or "").lower()
    return ("torrington" in t) or ("tdi" in t and "pricing" in t) or ("meridian" in t and "pricing" in t)

def parse_pdf_tdi(pdf_bytes: bytes) -> pd.DataFrame:
    """
    TDI Meridian / Spectrum (PDF):
    Extrae desde texto con pdfplumber:
      Part Number | Part Description | Seat Model | Aircraft | Leadtime | Unit | Sales Price | Min Buy
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Falta pdfplumber para leer PDF TDI: {e}")

    rows: List[Dict[str, Any]] = []
    pn_re = re.compile(r"^[A-Z0-9]+(?:[-/][A-Z0-9]+)*$")

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if not page_text:
                continue
            for line in page_text.splitlines():
                s = (line or "").strip()
                if not s:
                    continue

                low = s.lower()
                # saltar títulos/cabeceras/notes
                if low.startswith("torrington") or "pricing" in low and "part number" in low:
                    continue
                if low.startswith("page ") or low.startswith("tdi standard"):
                    continue
                if low.startswith("part number") or low.startswith("part description"):
                    continue

                tokens = s.split()
                if len(tokens) < 6:
                    continue

                pn = tokens[0].strip()
                if not pn_re.match(pn):
                    continue

                # desde el final: min buy, price, unit
                min_buy = tokens[-1] if tokens[-1].isdigit() else None
                price_raw = tokens[-2] if len(tokens) >= 2 else None
                unit = tokens[-3] if len(tokens) >= 3 else None

                if min_buy is None:
                    continue
                if parse_price_value(price_raw) is None:
                    continue

                # leadtime: buscamos 'days' o 'weeks'
                lead_idx = None
                for i in range(len(tokens) - 4, 0, -1):
                    if tokens[i].lower() in ("days", "day", "weeks", "week"):
                        lead_idx = i
                        break

                leadtime = None
                seat_model = None
                aircraft = None

                if lead_idx is not None:
                    # ejemplo: "100 Business Days"
                    lead_start = max(1, lead_idx - 2)
                    leadtime = " ".join(tokens[lead_start:lead_idx + 1]).strip()

                    pre = tokens[:lead_start]  # pn + desc + seat/aircraft...
                    if len(pre) >= 3:
                        seat_model = pre[-2]
                        aircraft = pre[-1]
                        desc_tokens = pre[1:-2]
                    elif len(pre) == 2:
                        desc_tokens = pre[1:]
                    else:
                        desc_tokens = []
                else:
                    # fallback: asumimos que el texto entre PN y los 3 últimos tokens es descripción
                    desc_tokens = tokens[1:-3]

                description = " ".join(desc_tokens).strip()

                rows.append({
                    "Part Number": pn,
                    "Part Description": description,
                    "Seat Model": seat_model,
                    "Aircraft": aircraft,
                    "Leadtime": leadtime,
                    "Unit": unit,
                    "Sales Price": price_raw,
                    "Min Buy": min_buy,
                })

    return pd.DataFrame(rows)


# ============================================================
#  Helpers para precios por tramos
# ============================================================
def parse_price_value(price_value) -> Optional[float]:
    """
    Convierte un valor de celda a float.
    Soporta formatos con €, $, comas, etc.
    Ignora textos como 'Price and Leadtime on request.'.
    """
    if price_value is None:
        return None

    s = str(price_value).strip()
    if not s:
        return None

    # Caso texto especial (AIRTEC, etc.)
    if s.lower().startswith("price and leadtime"):
        return None

    # quitar símbolos y espacios
    cleaned = (
        s.replace("€", "")
         .replace("$", "")
         .replace("£", "")
         .replace("¥", "")
         .replace(" ", "")
    )
    if not cleaned:
        return None

    # Manejar combinaciones de . y , (formato europeo vs US)
    s = cleaned
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        # Si la coma está más a la derecha, asumimos que es separador decimal
        if last_comma > last_dot:
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        # Solo comas: si parece decimal tipo "23,67" -> 23.67
        last_comma = s.rfind(",")
        decimals = s[last_comma + 1:]
        if decimals.isdigit() and len(decimals) in (2, 3):
            s = s.replace(",", ".")
        else:
            # coma como separador de miles
            s = s.replace(",", "")

    try:
        return float(s)
    except ValueError:
        return None


def parse_qty_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Convierte textos tipo '25-99', '100-249', '500-999', '>1000'
    en (min_qty, max_qty).
    """
    if text is None:
        return (None, None)

    t = str(text).strip().replace(" ", "")
    if not t:
        return (None, None)

    # >1000  -> (1001, None)
    if t.startswith(">"):
        nums = re.findall(r"\d+", t)
        if nums:
            return (int(nums[0]) + 1, None)
        return (None, None)

    # 25-99
    m = re.match(r"(\d+)-(\d+)", t)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # número suelto
    if t.isdigit():
        n = int(t)
        return (n, n)

    return (None, None)


def detect_header_and_qty_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Detecta la fila de encabezados y rangos de cantidad.
    Ajusta el DataFrame para que solo contenga datos y actualiza las columnas.
    """
    header_row_idx = None
    for i, row in df.head(40).iterrows():
        lower_vals = [str(v).strip().lower() for v in row.values if isinstance(v, str)]
        has_part_header = any("part" in v and "number" in v for v in lower_vals)
        has_description = "description" in lower_vals
        has_item = any(v.startswith("item") for v in lower_vals)
        if has_part_header or has_description or has_item:
            header_row_idx = i
            break

    qty_ranges_by_col_lower: Dict[str, str] = {}
    start_data_idx = 0
    col_names: List[str] = []

    if header_row_idx is not None:
        header_row = df.iloc[header_row_idx]
        next_row = df.iloc[header_row_idx + 1] if header_row_idx + 1 < len(df) else None

        for idx, val in enumerate(header_row):
            # Normalizar encabezados numéricos (ej: 500.0 -> 500)
            if val is not None and isinstance(val, (int, float)) and not pd.isna(val):
                if float(val).is_integer():
                    raw_name = str(int(val))
                else:
                    raw_name = str(val).strip()
            else:
                raw_name = str(val).strip() if val is not None else ""
            if not raw_name:
                raw_name = f"col_{idx}"

            col_name = raw_name
            col_key_lower = raw_name.strip().lower()

            # Para columnas Qty/ea, añadimos el rango de la fila siguiente
            if next_row is not None and "qty/ea" in col_key_lower:
                vr = next_row.iloc[idx]
                if isinstance(vr, str) and vr.strip():
                    range_text = vr.strip()
                    col_name = f"{raw_name} {range_text}"  # p.ej. "Qty/ea 25-99"
                    col_key_lower = col_name.strip().lower()
                    qty_ranges_by_col_lower[col_key_lower] = range_text

            col_names.append(col_name)

        # Si detectamos rangos, los datos empiezan 2 filas después;
        # si no, justo debajo de la cabecera.
        start_data_idx = header_row_idx + 1
        if qty_ranges_by_col_lower and next_row is not None:
            start_data_idx = header_row_idx + 2

        df = df.iloc[start_data_idx:].reset_index(drop=True)
        df.columns = col_names

    return df, qty_ranges_by_col_lower

def smart_prefix(term: str, min_len: int = 6) -> str:
    """
    Devuelve un prefijo corto para reducir candidatos SQL.
    Ej: '890107XXX' -> '890107'
    """
    if not term:
        return ""
    t = term.strip()
    return t if len(t) <= min_len else t[:min_len]

# ============================================================
#  Helpers específicos HOLMCO
# ============================================================
def parse_holmco_end_unit(raw: Optional[str]) -> List[str]:
    """
    Recibe algo como:
      '1046GT2102XX (91-06-05362)'
      '89-01-07XXX, 89-01-(X)-12, 89-01-(X)-16, 89-01-(X)-18'
    y devuelve:
      ['1046GT2102XX', '91-06-05362']
      ['89-01-07XXX', '89-01-(X)-12', '89-01-(X)-16', '89-01-(X)-18']
    """
    if not raw:
        return []

    # separar por coma o salto de línea
    tokens = re.split(r"[,\\n]+", str(raw))
    codes: List[str] = []

    for t in tokens:
        t = t.strip()
        if not t:
            continue

        # Caso "1046GT2102XX (91-06-05362)"
        m = re.match(r"^\s*([^\s(]+)\s*\(([^)]+)\)\s*$", t)
        if m:
            codes.append(m.group(1).strip())
            codes.append(m.group(2).strip())
        else:
            # Caso "89-01-07XXX" o "89-01-(X)-16"
            codes.append(t)

    # quitar duplicados manteniendo el orden
    seen = set()
    out: List[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def import_holmco_price_list(
    file_bytes: bytes,
    db: Session,
    supplier: models.Supplier,
    catalog: models.Catalog,
    default_currency: str,
) -> int:
    """
    Importa 'HOLMCO Price_List_2023_Issue1_Nov22 - USD.xlsx'.
    Por cada PART NUMBER crea:
      - Part (con precio base y min_qty_default = PRICE BREAK)
      - PriceTier
      - PartAlias para todos los códigos de END-UNIT
      - PartAttribute con todas las columnas extra.
    """
    # leer sin encabezado para encontrar la fila de PART NUMBER
    df_raw = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=None)

    header_idx = None
    for i, row in df_raw.iterrows():
        for cell in row:
            if isinstance(cell, str) and "part number" in cell.lower():
                header_idx = i
                break
        if header_idx is not None:
            break

    if header_idx is None:
        raise HTTPException(
            status_code=400,
            detail="No se encontró la fila de encabezados (PART NUMBER) en el catálogo HOLMCO.",
        )

    # Volver a leer usando esa fila como encabezado
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0, header=header_idx)

    # Mapear nombres de columnas
    col_map: Dict[str, str] = {}
    for col in df.columns:
        name = str(col).strip()
        lower = name.lower()

        if "part number" in lower:
            col_map[col] = "part_number"
        elif "description" in lower:
            col_map[col] = "description"
        elif "end-unit" in lower or "end unit" in lower:
            col_map[col] = "end_unit"
        elif "price usd" in lower or ("price" in lower and "usd" in lower):
            col_map[col] = "price_usd"
        elif "price break" in lower:
            col_map[col] = "price_break_pcs"
        elif "package" in lower and "qty" in lower:
            col_map[col] = "package_qty"
        elif "remark" in lower:
            col_map[col] = "remark"
        elif "lead time" in lower:
            col_map[col] = "lead_time"

    df = df.rename(columns=col_map)

    if "part_number" not in df.columns:
        raise HTTPException(
            status_code=400,
            detail="No se encontró la columna PART NUMBER en el catálogo HOLMCO.",
        )

    # Nos quedamos solo con filas que tienen PART NUMBER
    df = df[df["part_number"].notna()]
    df = df.where(pd.notnull(df), None)

    inserted = 0

    for _, row in df.iterrows():
        raw_code = row.get("part_number")
        if not raw_code:
            continue

        part_number_full, part_number_root = normalize_part_number(str(raw_code))

        description_val = row.get("description") or ""
        description = str(description_val).strip()

        end_unit_raw = row.get("end_unit") or ""

        # PRECIO
        price = parse_price_value(row.get("price_usd"))

        # PRICE BREAK -> min_qty_default
        price_break_val = row.get("price_break_pcs")
        min_qty_default = 1
        if price_break_val is not None:
            try:
                min_qty_default = int(str(price_break_val).strip())
            except ValueError:
                m = re.search(r"\d+", str(price_break_val))
                if m:
                    min_qty_default = int(m.group(0))

        # Crear Part
        part = models.Part(
            catalog_id=catalog.id,
            supplier_id=supplier.id,
            part_number_full=part_number_full,
            part_number_root=part_number_root,
            description=description,
            currency=default_currency,
            base_price=price,
            min_qty_default=min_qty_default,
        )
        db.add(part)
        db.flush()  # para obtener part.id
        inserted += 1

        # PriceTier
        if price is not None:
            pt = models.PriceTier(
                part_id=part.id,
                min_qty=min_qty_default,
                max_qty=None,
                unit_price=price,
                currency=default_currency,
            )
            db.add(pt)

        # Alias desde END-UNIT
        alias_codes = parse_holmco_end_unit(end_unit_raw)
        for code in alias_codes:
            db.add(
                models.PartAlias(
                    part_id=part.id,
                    code=code,
                    source="HOLMCO_END_UNIT",
                )
            )

        # Atributos extra: guardamos TODAS las columnas para este partnumber
        extra_cols = ["end_unit", "price_break_pcs", "package_qty", "remark", "lead_time"]
        for col_name in extra_cols:
            val = row.get(col_name)
            if val is None:
                continue
            attr = models.PartAttribute(
                part_id=part.id,
                attr_name=col_name,
                attr_value=str(val),
            )
            db.add(attr)

    db.commit()
    return inserted


# ============================================================
#  Helpers GMI AERO V2 (Corregido y Limpio)
# ============================================================

def clean_gmi_price(value):
    """Limpia y convierte precios, manejando nulos y texto."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    
    # Caso especial GMI: ignorar textos que no son precios
    if any(x in s.lower() for x in ["check", "request", "call", "specify"]):
        return None

    # Limpieza de moneda
    s = s.replace("€", "").replace("$", "").strip()
    
    # Manejo de formato europeo (1.200,00 -> 1200.00) vs US (1,200.00 -> 1200.00)
    # Si hay punto y coma, asumimos que la coma es decimal si está al final
    if "." in s and "," in s:
        last_dot = s.rfind(".")
        last_comma = s.rfind(",")
        if last_comma > last_dot: # Formato 1.500,00
            s = s.replace(".", "").replace(",", ".")
        else: # Formato 1,500.00
            s = s.replace(",", "")
    elif "," in s:
        # Solo comas: si parece decimal (2 decimales al final), reemplazar por punto
        # Si es 1,200 se asume mil.
        parts = s.split(",")
        if len(parts[-1]) == 2: # probable decimal "12,50"
             s = s.replace(",", ".")
        else:
             s = s.replace(",", "")
             
    try:
        return float(s)
    except ValueError:
        return None

def clean_gmi_part_number(value):
    """
    SOLUCIÓN PROBLEMA MISC:
    Elimina comillas, saltos de línea y espacios ocultos que corrompen la búsqueda.
    """
    if pd.isna(value):
        return None
    s = str(value)
    
    # 1. Eliminar comillas simples y dobles (causa del problema GMISAP021)
    s = s.replace('"', '').replace("'", "")
    
    # 2. Eliminar saltos de línea y retornos de carro
    s = s.replace('\n', '').replace('\r', '')
    
    # 3. Eliminar espacios duros (non-breaking space) y tabulaciones
    s = s.replace('\xa0', '').replace('\t', '')
    
    # 4. Trim final
    s = s.strip()
    
    if not s or s.lower() in ["nan", "n/a", "null"]:
        return None
        
    return s

def extract_gmi_parts_from_row(row, map_config, sheet_name, current_section, catalog_id, supplier_id, default_currency):
    """
    Extrae partes de una fila limpiando rigurosamente los códigos.
    Además:
      - Adjunta atributos extra (CM/Zones/etc.)
      - Devuelve ambos precios (Customer/Representative) si existen
    """
    parts_found = []

    idx = map_config.get("pn_idx")
    if idx is None or idx >= len(row):
        return []

    raw_pn = row[idx]
    s_pn = clean_gmi_part_number(raw_pn)

    if not s_pn:
        return []

    if s_pn.lower() in ["p/n", "pn", "part number", "item", "model", "description"]:
        return []

    if len(s_pn) > 25 and " " in s_pn and not any(c.isdigit() for c in s_pn):
        return []

    # Split por saltos de línea / "or"
    original_str = str(raw_pn).replace("\r", "\n")
    candidates = original_str.split("\n")

    final_pns = []
    for c in candidates:
        clean_c = clean_gmi_part_number(c)
        if not clean_c:
            continue

        if " or " in clean_c.lower():
            for sp in re.split(r"\bor\b", clean_c, flags=re.IGNORECASE):
                sp = sp.strip(" ,;")
                if sp:
                    final_pns.append(clean_gmi_part_number(sp.upper()))
        else:
            final_pns.append(clean_c)

    # Precios
    p_cust = None
    if map_config.get("price_cust_idx") is not None and map_config["price_cust_idx"] < len(row):
        p_cust = clean_gmi_price(row[map_config["price_cust_idx"]])

    p_rep = None
    if map_config.get("price_rep_idx") is not None and map_config["price_rep_idx"] < len(row):
        p_rep = clean_gmi_price(row[map_config["price_rep_idx"]])

    base_price = p_cust if p_cust is not None else p_rep

    # Descripción
    desc = ""
    if map_config.get("desc_idx") is not None and map_config["desc_idx"] < len(row):
        d_val = row[map_config["desc_idx"]]
        if pd.notna(d_val):
            desc = str(d_val).strip().replace("\n", " ")

    # Atributos extra (CM/Zones/etc.)
    extra_attributes = []
    for col_idx, label in map_config.get("extra_desc_idxs", []):
        if col_idx < len(row):
            v = row[col_idx]
            if pd.notna(v):
                vv = str(v).strip()
                if vv and vv.lower() != "nan":
                    lab = str(label).replace("\n", " ").strip()
                    name = lab.upper() if lab.lower() in ["cm", "mm"] else lab.title()
                    extra_attributes.append({"name": name, "value": vv})

    # Guardar ambos precios como atributos (para que el front los muestre sí o sí)
    if p_cust is not None:
        extra_attributes.append({"name": "Unit Price (Customer)", "value": str(p_cust)})
    if p_rep is not None:
        extra_attributes.append({"name": "Unit Price (Representative)", "value": str(p_rep)})

    for pn in final_pns:
        if not pn:
            continue

        if "(" in pn:
            pn = pn.split("(")[0].strip()

        parts_found.append({
            "catalog_id": catalog_id,
            "supplier_id": supplier_id,
            "part_number_full": pn,
            "part_number_root": pn.split("-")[0],
            "description": desc,
            "currency": default_currency,
            "base_price": base_price,
            "section": current_section,
            "sheet": sheet_name,
            "price_rep": p_rep,
            "extra_attributes": extra_attributes,
        })

    return parts_found


def process_gmi_sheet_v2(
    db: Session,
    catalog: models.Catalog,
    supplier: models.Supplier,
    df_raw: pd.DataFrame,
    sheet_name: str,
    default_currency: str
) -> int:
    inserted_count = 0
    active_maps = []
    current_section = sheet_name

    # Convertimos todo a lista de listas para máxima velocidad
    rows = df_raw.values.tolist()
    
    # Palabras clave para detectar headers explícitos
    header_keywords = ["p/n", "pn", "part number", "part_number", "model"]

    def _auto_discover_map(row_data, pn_col_idx):
        """
        PLAN B: Si encontramos un código GMI pero no tenemos mapa para esa columna,
        deducimos dónde están el precio y la descripción analizando los tipos de datos de la fila.
        """
        map_cfg = {"pn_idx": pn_col_idx, "extra_desc_idxs": []}
        
        # 1. Buscar Descripción (Texto largo a la derecha)
        # 2. Buscar Precio (Número o moneda a la derecha)
        limit_search = min(len(row_data), pn_col_idx + 8) # Mirar hasta 8 columnas adelante
        
        prices_found = []
        desc_found = None
        
        for i in range(pn_col_idx + 1, limit_search):
            val = row_data[i]
            val_clean = str(val).strip()
            if not val_clean or val_clean.lower() == "nan":
                continue
                
            # ¿Es Precio?
            price_candidate = clean_gmi_price(val)
            if price_candidate is not None:
                prices_found.append(i)
                continue
                
            # ¿Es Descripción? (Texto largo, no numérico, no fecha)
            # Evitamos celdas cortas que puedan ser unidades ("EA", "m")
            if len(val_clean) > 3 and desc_found is None:
                desc_found = i
            elif any(x in val_clean.lower() for x in ["cm", "mm", "diam", "zone", "size"]):
                # Atributos tipo "Dimensions"
                map_cfg["extra_desc_idxs"].append((i, "Attr"))

        # Asignar lo encontrado
        if desc_found is not None:
            map_cfg["desc_idx"] = desc_found
        
        if prices_found:
            map_cfg["price_cust_idx"] = prices_found[0] # El primero suele ser Customer
            if len(prices_found) > 1:
                map_cfg["price_rep_idx"] = prices_found[1] # El segundo Representative

        # 3. Buscar Atributos a la IZQUIERDA (ID, Item en Consumables)
        # Miramos 2 columnas atrás
        start_left = max(0, pn_col_idx - 2)
        for i in range(start_left, pn_col_idx):
            val = row_data[i]
            if val and str(val).strip() and str(val).lower() != "nan":
                # Si hay algo escrito a la izquierda, lo agregamos como "Info"
                map_cfg["extra_desc_idxs"].insert(0, (i, "Info"))

        return map_cfg

    for row_idx, row in enumerate(rows):
        row_str_list = [str(c).strip().lower() for c in row]
        
        # --- PASO 1: Análisis de Datos (Data Scan) ---
        # Identificamos qué columnas tienen códigos GMI válidos en esta fila
        gmi_cols_indices = []
        for i, val in enumerate(row):
            val_clean = clean_gmi_part_number(val)
            if val_clean and str(val_clean).upper().startswith("GMI"):
                gmi_cols_indices.append(i)

        # --- PASO 2: Gestión de Mapas (Header Detection & Fusion) ---
        # Si NO hay datos GMI, buscamos headers para preparar las siguientes filas
        if not gmi_cols_indices:
            pn_indices = [i for i, val in enumerate(row_str_list) if val in header_keywords]
            if pn_indices:
                # Detectamos nuevos headers
                new_maps = []
                for start_idx in pn_indices:
                    # Lógica estándar de mapeo por headers (como en versiones anteriores)
                    map_cfg = {"pn_idx": start_idx, "extra_desc_idxs": []}
                    
                    # Límite derecho para no invadir
                    next_pn = 9999
                    for other in pn_indices: 
                        if other > start_idx: next_pn = min(next_pn, other)
                    limit = min(len(row), next_pn)
                    
                    # Buscar columnas por nombre
                    unit_cols = []
                    for i in range(start_idx + 1, limit):
                        v = row_str_list[i]
                        if "desc" in v: map_cfg["desc_idx"] = i
                        if any(k in v for k in ["cm", "dim", "zone", "size", "diam"]):
                            map_cfg["extra_desc_idxs"].append((i, v.title()))
                        if "customer" in v: map_cfg["price_cust_idx"] = i
                        if "representative" in v: map_cfg["price_rep_idx"] = i
                        if "price" in v or "unit" in v or "€" in v: unit_cols.append(i)
                    
                    # Fallback precios
                    if "price_cust_idx" not in map_cfg and unit_cols: map_cfg["price_cust_idx"] = unit_cols[0]
                    if "price_rep_idx" not in map_cfg and len(unit_cols) > 1: map_cfg["price_rep_idx"] = unit_cols[1]
                    
                    # Mirar izquierda (ID, Type)
                    l_limit = 0
                    for other in pn_indices: 
                        if other < start_idx: l_limit = max(l_limit, other+1)
                    for i in range(l_limit, start_idx):
                        if row_str_list[i] in ["id", "type", "category"]:
                            map_cfg["extra_desc_idxs"].insert(0, (i, row_str_list[i].title()))
                            
                    new_maps.append(map_cfg)
                
                # Reemplazamos mapas activos si es una fila puramente de encabezados
                active_maps = new_maps
            continue # Saltamos fila de solo headers

        # --- PASO 3: Extracción y Auto-Descubrimiento (La Magia) ---
        # Iteramos sobre las columnas donde ENCONTRAMOS datos GMI
        for gmi_col in gmi_cols_indices:
            
            # A) ¿Tenemos un mapa activo para esta columna?
            current_map = next((m for m in active_maps if m["pn_idx"] == gmi_col), None)
            
            # B) Si NO tenemos mapa (tabla nueva sin header, o header perdido), ¡LO CREAMOS!
            if not current_map:
                # 
                current_map = _auto_discover_map(row, gmi_col)
                # Lo agregamos a activos para que sirva para las siguientes filas también
                active_maps.append(current_map)

            # C) Extraer datos usando el mapa (existente o descubierto)
            # Usamos una sección genérica si no hay título detectado
            parts = extract_gmi_parts_from_row(
                row, current_map, sheet_name, current_section,
                catalog.id, supplier.id, default_currency
            )
            
            for p_data in parts:
                part = models.Part(
                    catalog_id=p_data["catalog_id"],
                    supplier_id=p_data["supplier_id"],
                    part_number_full=p_data["part_number_full"],
                    part_number_root=p_data["part_number_root"],
                    description=p_data["description"],
                    currency=p_data["currency"],
                    base_price=p_data["base_price"],
                    min_qty_default=1
                )
                db.add(part)
                db.flush()
                inserted_count += 1
                
                if p_data["base_price"] is not None:
                    db.add(models.PriceTier(
                        part_id=part.id, min_qty=1,
                        unit_price=p_data["base_price"], currency=p_data["currency"]
                    ))
                
                # Atributos
                db.add(models.PartAttribute(part_id=part.id, attr_name="Sección", attr_value=p_data["section"]))
                db.add(models.PartAttribute(part_id=part.id, attr_name="Hoja Original", attr_value=p_data["sheet"]))
                for a in p_data.get("extra_attributes", []):
                    db.add(models.PartAttribute(part_id=part.id, attr_name=a["name"], attr_value=a["value"]))
        
        # --- PASO 4: Detectar Títulos de Sección (Opcional) ---
        # Si la fila tiene texto en col 0 pero NO es GMI, podría ser un título nuevo
        if 0 not in gmi_cols_indices and isinstance(row[0], str):
            clean_txt = str(row[0]).strip()
            # Si es texto largo y no es header
            if len(clean_txt) > 3 and clean_txt.lower() not in header_keywords:
                 current_section = clean_txt

    return inserted_count
def import_gmi_catalog(
    file_bytes: bytes,
    db: Session,
    supplier: models.Supplier,
    catalog: models.Catalog,
    default_currency: str
) -> int:
    try:
        xl = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
    except Exception:
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        
    total_inserted = 0
    
    for sheet_name in xl.sheet_names:
        sn_lower = sheet_name.lower()
        if "read me" in sn_lower or "terms" in sn_lower or "cover" in sn_lower:
            continue
            
        # Leer sin header para procesar manualmente
        df_raw = xl.parse(sheet_name, header=None)
        if df_raw.empty: continue
            
        count = process_gmi_sheet_v2(db, catalog, supplier, df_raw, sheet_name, default_currency)
        total_inserted += count
        
    db.commit()
    return total_inserted

# ============================================================
#  Endpoint: subir catálogo (multi-hoja, Excel/CSV/PDF)
# ============================================================
@app.post("/catalogs/upload")
async def upload_catalog(
    supplier_name: str = Form(...),
    year: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content = await file.read()
    ext = os.path.splitext(file.filename)[1].lower()

    # moneda por defecto deducida del nombre del archivo
    default_currency = "EUR" if "eur" in file.filename.lower() else "USD"

    if "holmco" in supplier_name.lower() or "holmco" in file.filename.lower():
        supplier = get_or_create_supplier(db, supplier_name)
        catalog = create_catalog(db, supplier, year, file.filename)

        inserted = import_holmco_price_list(
            file_bytes=content,
            db=db,
            supplier=supplier,
            catalog=catalog,
            default_currency=default_currency,
        )

        return {
            "message": f"Catálogo HOLMCO cargado (piezas únicas insertadas: {inserted})",
            "catalog_id": catalog.id,
        }
        # --- NUEVO BLOQUE GMI AERO ---
    if "gmi" in supplier_name.lower() or "gmi" in file.filename.lower():
        supplier = get_or_create_supplier(db, supplier_name)
        catalog = create_catalog(db, supplier, year, file.filename)

        inserted = import_gmi_catalog(
            file_bytes=content,
            db=db,
            supplier=supplier,
            catalog=catalog,
            default_currency=default_currency,
        )

        return {
            "message": f"Catálogo GMI procesado. Hojas escaneadas. (Piezas insertadas: {inserted})",
            "catalog_id": catalog.id,
        }

    # ===============================
    # 1) LEER TODAS LAS HOJAS / TABLAS EN DATAFRAMES
    # ===============================
    dfs: List[pd.DataFrame] = []

    try:
        if ext in (".xlsx", ".xls"):
            try:
                excel_file = pd.ExcelFile(
                    io.BytesIO(content),
                    engine="openpyxl" if ext == ".xlsx" else None,
                )
                for sheet_name in excel_file.sheet_names:
                    tmp_df = excel_file.parse(sheet_name)
                    if not tmp_df.empty:
                        dfs.append(tmp_df)
            except Exception as e_openpyxl:
                # Fallback solo para .xlsx (primera hoja)
                if ext == ".xlsx":
                    print("[upload_catalog] openpyxl falló, usando fallback XML:", e_openpyxl)
                    dfs = [read_xlsx_fallback(content)]
                else:
                    raise

        elif ext == ".csv":
            try:
                dfs = [pd.read_csv(io.BytesIO(content))]
            except Exception:
                dfs = [pd.read_csv(io.BytesIO(content), sep=None, engine="python")]

        elif ext == ".pdf":
            # PDFs: algunos son tabulares (pdfplumber) y otros requieren parseo por texto (pypdf)
            dfs = read_pdf_vendor_dfs(content)

        else:
            raise HTTPException(
                status_code=400,
                detail="Por ahora solo se aceptan archivos Excel (.xls, .xlsx), CSV o PDFs tabulares.",
            )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "No se pudo leer el archivo. "
                "Es posible que tenga un formato no compatible. "
                f"Detalle técnico: {e}"
            ),
        )

    if not dfs or all(df.empty for df in dfs):
        raise HTTPException(status_code=400, detail="El archivo no contiene datos reconocibles.")

    # Creamos supplier y catálogo una sola vez para todas las hojas
    supplier = get_or_create_supplier(db, supplier_name)
    catalog = create_catalog(db, supplier, year, file.filename)

    # cache para no crear varios Part con el mismo código
    # clave: (catalog_id, supplier_id, part_number_full)
    part_cache: Dict[Tuple[int, int, str], models.Part] = {}
    inserted = 0

    # ===============================
    # Procesar cada hoja / tabla
    # ===============================
    for original_df in dfs:
        if original_df.empty:
            continue

        # 2) Detectar encabezados y rangos Qty de ESTA hoja
        df, qty_ranges_by_col_lower = detect_header_and_qty_ranges(original_df)
        if df.empty:
            continue

        # ===============================
        # 3) NORMALIZAR NOMBRES DE COLUMNAS
        # ===============================
        column_map = {
            # códigos de pieza
            "part number": "part_number", "part_number": "part_number", "part numbep": "part_number",
            "pn": "part_number", "p/n": "part_number", "part-no.": "part_number",
            "part-no": "part_number", "part no.": "part_number",

            # Article / ArticleNo
            "articleno.": "article_no", "article no.": "article_no", "article no": "article_no",

            # columnas ITEM (AIRTEC-BRAIDS)
            "item (standard)": "part_number", "item (other)": "part_number", "item": "part_number",


            # --- NUEVOS PDFs / proveedores ---
            # IPECO (PDF)
            "material": "part_number",
            "price (usd)": "price",
            "uom": "unit_code", "uom.": "unit_code", "unit of measure": "unit_code",
            "lead time (days)": "lead_time",
            "of": "package_qty", "per": "per_qty",

            # DIEHL (PDF)
            "pnr": "part_number",
            "price 2026": "price",
            "cur": "currency",
            "unt": "unit_code",
            "moq": "min_qty",
            "spq": "package_qty",
            "ltm": "lead_time",

            # COLLINS / equivalencias
            "standard part number": "standard_part_number",
            "rf part number": "rf_part_number",
            "rf partnumber": "rf_part_number",
            "rf part no": "rf_part_number",

            # descripción
            "descripcion": "description", "description": "description",

            # precio
            "price": "price", "precio": "price", "master $": "price",

            # moneda
            "currency": "currency", "moneda": "currency",

            # cantidades mínimas / máximas
            "min qty": "min_qty", "min_qty": "min_qty", "from qty": "min_qty",
            "to qty": "max_qty", "max qty": "max_qty",
            "quantity": "min_qty", "qty": "min_qty",


            # --- NUEVOS XLSX (PUROLATOR / SAFRAN) ---
            "partnumber": "part_number",
            "lead-time": "lead_time", "leadtime": "lead_time",
            "stock ref": "stock_ref", "stock ref.": "stock_ref",
            "safran stock ref.": "stock_ref",
            "comments": "comments", "comment": "comments", "remarks": "comments", "remark": "comments",
            "uon": "unit_code", "u/o/m": "unit_code", "uom": "unit_code", "uom.": "unit_code",

            # --- NUEVOS PDF (TDI) ---
            "part description": "description",
            "seat model": "seat_model",
            "aircraft": "aircraft",
            "manufacturing": "manufacturing",
            "leadtime": "lead_time",
            "sales price": "price",
            "min buy": "min_qty",

            # otros posibles
            "unit code": "unit_code", "unit": "unit_code", "lead time": "lead_time",
            "lifecycle": "lifecycle", "cumulative": "cumulative",
        }

        normalized_cols: Dict[str, str] = {}
        for col in df.columns:
            raw_name = str(col).strip()
            col_key = raw_name.lower()

            # 1) Mapeo exacto
            if col_key in column_map:
                mapped = column_map[col_key]

            # 2) PANASONIC Master Price List: columnas tipo "Master S2 2023", etc.
            elif "master" in col_key and ("s2" in col_key or "$" in col_key):
                mapped = "price"

            # 3) Variaciones de acumulado / lifecycle
            elif "cumulative" in col_key:
                mapped = "cumulative"
            elif "lifecycle" in col_key:
                mapped = "lifecycle"

            else:
                mapped = col_key

            normalized_cols[col] = mapped

        df.rename(columns=normalized_cols, inplace=True)

        # Evitar columnas duplicadas (ej: PUROLATOR trae dos 'Part Number')
        df = dedupe_columns(df)

        # 3.1 Para catálogos tipo STUKERJURGEN:
        if "article_no" in df.columns:
            if "part_number" in df.columns:
                df["supplier_part_no"] = df["part_number"]
            df["part_number"] = df["article_no"]

        # Rellenar hacia abajo códigos que vienen en blanco en filas hijas
        if "part_number" in df.columns:
            df["part_number"] = df["part_number"].replace("", None).ffill()
        if "article_no" in df.columns:
            df["article_no"] = df["article_no"].replace("", None).ffill()
        if "rf_part_number" in df.columns:
            df["rf_part_number"] = df["rf_part_number"].replace("", None).ffill()
        if "standard_part_number" in df.columns:
            df["standard_part_number"] = df["standard_part_number"].replace("", None).ffill()
        if "supplier_part_no" in df.columns:
            df["supplier_part_no"] = df["supplier_part_no"].replace("", None).ffill()

        # Columnas de precios por tramo:
        #  - AIRTEC-BRAIDS: "Qty/ea" + rango en fila siguiente (qty_ranges_by_col_lower)
        #  - PUROLATOR: rangos/números en los encabezados (5-9, 10-24, 500, 1000, ...)
        tier_specs: List[Dict[str, Optional[int]]] = []
        tier_col_names: set = set()

        # A) Caso qty_ranges_by_col_lower (Qty/ea + fila de rangos)
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in qty_ranges_by_col_lower:
                range_text = qty_ranges_by_col_lower[col_lower]
                min_q, max_q = parse_qty_range(range_text)
                tier_specs.append({"col": col, "min_qty": min_q, "max_qty": max_q})
                tier_col_names.add(str(col))

        # B) Caso encabezados que SON el tramo (5-9 / 10 - 24 / 500 / 1000 ...)
        for col in df.columns:
            col_name = col
            min_q, max_q = parse_tier_header(col_name)
            if min_q is None:
                continue
            # evitamos duplicar si ya viene del caso A)
            if str(col_name) in tier_col_names:
                continue
            tier_specs.append({"col": col_name, "min_qty": min_q, "max_qty": max_q})
            tier_col_names.add(str(col_name))

        # ===============================
        # 4) RECORRER FILAS E INSERTAR EN BD
        # ===============================
        last_part_code: Optional[str] = None  # para reutilizar ArticleNo en filas sin código

        for _, row in df.iterrows():
            # ---------- CÓDIGO DE PARTE ----------
            candidates = [
                # Prioridad: RF -> Part Number -> Standard -> Article
                row.get("rf_part_number"),
                row.get("part_number"),
                row.get("standard_part_number"),
                row.get("supplier_part_no"),
                row.get("article_no"),
                row.get("part number"),
                row.get("pn"),
            ]
            raw_code = None
            for c in candidates:
                if c is not None and str(c).strip() != "":
                    raw_code = c
                    break

            # Si esta fila no trae código, usamos el último
            if (raw_code is None or str(raw_code).strip() == "") and last_part_code is not None:
                raw_code = last_part_code

            # Si sigue sin código, no podemos usar esta fila
            if raw_code is None or str(raw_code).strip() == "":
                continue

            part_number_full, part_number_root = normalize_part_number(str(raw_code))

            # Si el "código" no tiene ningún dígito (fila de encabezado), ignorar
            if not any(ch.isdigit() for ch in part_number_root):
                continue

            # Guardamos el último código visto
            last_part_code = raw_code

            # ---------- DESCRIPCIÓN ----------
            description_val = row.get("description", "")
            description = (
                str(description_val)
                if description_val is not None and str(description_val) != "nan"
                else ""
            )

            # ---------- MONEDA ----------
            currency_val = row.get("currency", None)
            currency = (
                str(currency_val).upper()
                if currency_val is not None and str(currency_val) != "nan"
                else None
            )

            # ---------- CANTIDAD MÍNIMA ----------
            min_qty = row.get("min_qty")

            # Por defecto 1 unidad
            min_qty_default = 1
            if min_qty is not None and str(min_qty) != "nan":
                s_min = str(min_qty).strip()
                m_qty = re.search(r"\d+", s_min)
                if m_qty:
                    min_qty_default = int(m_qty.group(0))

            # Si MOQ viene en comentarios (p.ej. SAFRAN: "MOQ 100 EA")
            if min_qty_default == 1:
                moq_from_comments = parse_moq_from_text(row.get("comments"))
                if moq_from_comments is not None:
                    min_qty_default = moq_from_comments

            # ---------------- PRECIOS ----------------
            # 1) Precio base desde una columna estándar "price"
            base_price = parse_price_value(row.get("price"))

            # 2) Precios por tramo desde columnas tipo "qty/ea 25-99", ...
            tier_prices: List[Tuple[Optional[int], Optional[int], float]] = []
            pricing_note: Optional[str] = None

            for spec in tier_specs:
                col_name = spec["col"]
                if col_name not in row:
                    continue
                raw_val = row[col_name]
                if raw_val is None or str(raw_val) == "nan":
                    continue

                unit_price = parse_price_value(raw_val)
                if unit_price is None:
                    # Caso "Price and Leadtime on request."
                    if isinstance(raw_val, str) and raw_val.strip():
                        pricing_note = pricing_note or raw_val.strip()
                    continue

                min_q = spec["min_qty"] or min_qty_default
                max_q = spec["max_qty"]
                tier_prices.append((min_q, max_q, unit_price))

            # Si tenemos precios por tramo pero no base_price, usamos el tramo de menor qty
            if tier_prices and base_price is None:
                tier_prices_sorted = sorted(
                    tier_prices,
                    key=lambda t: t[0] if t[0] is not None else 0
                )
                base_price = tier_prices_sorted[0][2]

            # --------- UNIFICAR PARTS POR CÓDIGO ---------
            cache_key = (catalog.id, supplier.id, part_number_full)
            part = part_cache.get(cache_key)

            is_new_part = False
            if part is None:
                # Creamos el Part solo la primera vez que vemos ese código
                part = models.Part(
                    catalog_id=catalog.id,
                    supplier_id=supplier.id,
                    part_number_full=part_number_full,
                    part_number_root=part_number_root,
                    description=description,
                    currency=currency or default_currency,
                    base_price=base_price,
                    min_qty_default=min_qty_default,
                )
                db.add(part)
                db.flush()
                part_cache[cache_key] = part
                inserted += 1
                is_new_part = True


            # --------- ALIASES / CÓDIGOS EQUIVALENTES ---------
            # Permite buscar por "standard part number" y encontrar el RF (y viceversa).
            if is_new_part:
                alias_values: List[Any] = []

                # campos conocidos (cuando existan)
                alias_values.extend([
                    row.get("standard_part_number"),
                    row.get("rf_part_number"),
                    row.get("supplier_part_no"),
                ])

                # si el part_number "visible" no fue el principal, lo agregamos como alias
                alias_values.append(row.get("part_number"))
                alias_values.append(row.get("article_no"))

                # columnas duplicadas de part number (PUROLATOR trae varias)
                for c_name in df.columns:
                    if str(c_name).startswith("part_number__"):
                        alias_values.append(row.get(c_name))

                # columnas genéricas que contengan la palabra "alias"/"alternate"
                for col_name, value in row.items():
                    norm_name = normalized_cols.get(col_name, str(col_name).strip().lower())
                    if any(k in norm_name for k in ("alias", "alternate", "equivalent", "equivalente", "alt " , " alt", "alt_pn")):
                        alias_values.append(value)

                main_norm = normalize_pn(part.part_number_full)
                seen_alias_norms: set = set()

                for raw_alias in alias_values:
                    for code_alias in extract_alias_codes(raw_alias):
                        if not code_alias:
                            continue
                        if normalize_pn(code_alias) == main_norm:
                            continue
                        norm = normalize_pn(code_alias)
                        if norm in seen_alias_norms:
                            continue
                        seen_alias_norms.add(norm)

                        db.add(
                            models.PartAlias(
                                part_id=part.id,
                                code=str(code_alias).strip(),
                                source="AUTO_ALIAS",
                            )
                        )

            # --------- PRICE TIERS (rangos / cantidades) ---------
            if tier_prices:
                # Catálogos tipo AIRTEC-BRAIDS con varias columnas Qty/ea
                for min_q, max_q, unit_price in tier_prices:
                    pt = models.PriceTier(
                        part_id=part.id,
                        min_qty=min_q if min_q is not None else min_qty_default,
                        max_qty=max_q,
                        unit_price=unit_price,
                        currency=currency or default_currency,
                    )
                    db.add(pt)
            else:
                # Catálogos tradicionales (STUKERJURGEN Hansair):
                # una fila por cada (qty, price).
                max_qty = row.get("max_qty")
                has_qty_info = (
                    (min_qty is not None and str(min_qty) != "nan")
                    or (max_qty is not None and str(max_qty) != "nan")
                )
                if has_qty_info or base_price is not None:
                    pt = models.PriceTier(
                        part_id=part.id,
                        min_qty=min_qty_default,
                        max_qty=(
                            int(max_qty)
                            if max_qty is not None and str(max_qty) != "nan"
                            else None
                        ),
                        unit_price=base_price if base_price is not None else 0.0,
                        currency=currency or default_currency,
                    )
                    db.add(pt)

            # Si sólo hay "Price and Leadtime on request." y nada numérico,
            # guardamos esa info como atributo.
            if pricing_note and not tier_prices and base_price is None:
                attr_note = models.PartAttribute(
                    part_id=part.id,
                    attr_name="pricing_note",
                    attr_value=pricing_note,
                )
                db.add(attr_note)

            # --------- ATRIBUTOS EXTRA ---------
            if is_new_part:
                standard = {
                    "part_number", "article_no", "description", "price",
                    "currency", "min_qty", "max_qty",
                }
                for col_name, value in row.items():
                    norm_name = normalized_cols.get(
                        col_name, str(col_name).strip().lower()
                    )
                    if norm_name in standard:
                        continue
                    if value is None or str(value) == "nan":
                        continue
                    if "qty/ea" in norm_name.lower():
                        continue
                    # Evitar guardar columnas de precios por tramo (p.ej. 5-9 / 10-24 / 500...)
                    if str(col_name) in tier_col_names:
                        continue
                    attr = models.PartAttribute(
                        part_id=part.id,
                        attr_name=norm_name,
                        attr_value=str(value),
                    )
                    db.add(attr)
    db.commit()
    return {
        "message": f"Catálogo cargado (piezas únicas insertadas: {inserted})",
        "catalog_id": catalog.id,
    }
# ============================================================
#  Endpoint: estadísticas simples (/stats)
# ============================================================
@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """
    Devuelve:
      - cantidad de catálogos cargados
      - cantidad de repuestos en BD
      - timestamp de generación
    """
    catalogs_count = db.query(models.Catalog).count()
    parts_count = db.query(models.Part).count()

    return {
        "catalogs": catalogs_count,
        "parts": parts_count,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }
# ============================================================
#  Endpoint: búsqueda de piezas (incluye nombre del proveedor)
#  - Soporta búsqueda sin guiones/espacios (normalización)
#  - Soporta comodín X / XX / XXX (match flexible)
# ============================================================


@app.get("/parts/search")
def search_parts(
    query: str = Query(
        ...,
        alias="q",
        min_length=1,
        description="Código o parte de la descripción (puede ser múltiple separado por coma o líneas)",
    ),
    db: Session = Depends(get_db),
):
    raw = (query or "").strip()
    if not raw:
        return []

    # términos separados por coma / salto de línea / ; / tab
    terms = [t.strip() for t in re.split(r"[,\n;\t]+", raw) if t.strip()]
    terms = terms[:30]

    groups = []
    for t in terms:
        t_norm = normalize_pn(t)  # <-- quita guiones/espacios/etc y uppercase

        like_prefix = f"{t}%"
        like_any = f"%{t}%"

        # versión normalizada para búsquedas sin guiones (89-01-07XXX == 890107XXX)
        like_prefix_norm = f"{t_norm}%"
        like_any_norm = f"%{t_norm}%"

        groups.append(
            or_(
                # ----------------------------
                # búsquedas "normales"
                # ----------------------------
                models.Part.part_number_full.ilike(like_prefix),
                models.Part.part_number_root.ilike(like_prefix),
                models.Part.description.ilike(like_any),
                models.PartAttribute.attr_value.ilike(like_any),
                models.PartAlias.code.ilike(like_any),

                # ----------------------------
                # búsquedas "normalizadas" (sin guiones)
                # ----------------------------
                sql_normalize(models.Part.part_number_full, db).ilike(like_prefix_norm),
                sql_normalize(models.Part.part_number_root, db).ilike(like_prefix_norm),
                sql_normalize(models.PartAlias.code, db).ilike(like_any_norm),
            )
        )

    combined_filter = or_(*groups)

    rows = (
        db.query(models.Part, models.Supplier)
        .join(models.Supplier, models.Part.supplier_id == models.Supplier.id)
        .outerjoin(models.PartAttribute, models.PartAttribute.part_id == models.Part.id)
        .outerjoin(models.PartAlias, models.PartAlias.part_id == models.Part.id)
        .filter(combined_filter)
        .order_by(models.Part.part_number_full)
        .limit(200)
        .all()
    )

    results = []
    seen_parts = set()

    def push_part(part, supplier):
        if part.id in seen_parts:
            return
        seen_parts.add(part.id)

        item = {
            "id": part.id,
            "part_number_full": part.part_number_full,
            "part_number_root": part.part_number_root,
            "description": part.description,
            "currency": part.currency,
            "base_price": part.base_price,
            "min_qty_default": part.min_qty_default,
            "catalog_id": part.catalog_id,
            "supplier_id": part.supplier_id,
            "supplier_name": supplier.name,
            "price_tiers": [
                {
                    "id": pt.id,
                    "min_qty": pt.min_qty,
                    "max_qty": pt.max_qty,
                    "unit_price": pt.unit_price,
                    "currency": pt.currency,
                }
                for pt in getattr(part, "price_tiers", [])
            ],
            "aliases": [
                {
                    "id": al.id,
                    "code": al.code,
                    "source": al.source,
                }
                for al in db.query(models.PartAlias).filter(models.PartAlias.part_id == part.id).all()
            ],
            "attributes": [
                {
                    "id": attr.id,
                    "attr_name": attr.attr_name,
                    "attr_value": attr.attr_value,
                }
                for attr in getattr(part, "attributes", [])
            ],
        }
        results.append(item)

    for part, supplier in rows:
        push_part(part, supplier)

    # =========================================================
    # ✅ SEGUNDA PASADA: Match contra patrones con 'X' (comodín)
    #   - Comparación usando normalize_pn() dentro de wildcard_x_match()
    #   - Considera también aliases con X
    # =========================================================
    if len(results) < 200:
        for t in terms:
            t_clean = (t or "").strip()
            if not t_clean:
                continue

            t_norm = normalize_pn(t_clean)

            # Prefijo corto para traer pocos candidatos
            pref = smart_prefix(t_norm, min_len=6)
            if not pref:
                continue

            wildcard_candidates = (
                db.query(models.Part, models.Supplier)
                .join(models.Supplier, models.Part.supplier_id == models.Supplier.id)
                .outerjoin(models.PartAlias, models.PartAlias.part_id == models.Part.id)
                .filter(
                    or_(
                        # Candidatos por part_number_full que tengan X (normalizado)
                        and_(
                            sql_normalize(models.Part.part_number_full, db).ilike(f"{pref}%"),
                            sql_normalize(models.Part.part_number_full, db).ilike("%X%"),
                        ),
                        # Candidatos por alias que tengan X (normalizado)
                        and_(
                            sql_normalize(models.PartAlias.code, db).ilike(f"{pref}%"),
                            sql_normalize(models.PartAlias.code, db).ilike("%X%"),
                        ),
                    )
                )
                .limit(800)
                .all()
            )

            for part, supplier in wildcard_candidates:
                if part.id in seen_parts:
                    continue

                # Match por PN principal (wildcard_x_match ya normaliza por dentro)
                ok = wildcard_x_match(part.part_number_full, t_clean)

                # Match por alias
                if not ok:
                    aliases = getattr(part, "aliases", None)
                    if aliases is not None:
                        ok = any(
                            wildcard_x_match(a.code, t_clean)
                            for a in aliases
                            if a and a.code
                        )
                    else:
                        alias_rows = (
                            db.query(models.PartAlias.code)
                            .filter(models.PartAlias.part_id == part.id)
                            .all()
                        )
                        ok = any(
                            wildcard_x_match(code, t_clean)
                            for (code,) in alias_rows
                            if code
                        )

                if ok:
                    push_part(part, supplier)
                    if len(results) >= 200:
                        break

            if len(results) >= 200:
                break

    return results


@app.get("/catalogs", response_model=List[schemas.CatalogListOut])
def list_catalogs(db: Session = Depends(get_db)):
    """
    Lista todos los catálogos ordenados por fecha de creación (más recientes primero).
    Incluye el nombre del proveedor haciendo un JOIN implícito o explícito.
    """
    # Obtenemos catálogos y cargamos la relación con Supplier
    catalogs = (
        db.query(models.Catalog)
        .join(models.Supplier)
        .order_by(models.Catalog.created_at.desc())
        .all()
    )
    
    # Formateamos la respuesta para que coincida con el esquema
    results = []
    for cat in catalogs:
        results.append({
            "id": cat.id,
            "supplier_name": cat.supplier.name,  # Accedemos a la relación
            "year": cat.year,
            "original_filename": cat.original_filename,
            "created_at": cat.created_at
        })
    
    return results

from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

@app.delete("/catalogs/{catalog_id}")
def delete_catalog(catalog_id: int, db: Session = Depends(get_db)):
    # 1) Verificar que exista
    catalog = db.query(models.Catalog).filter(models.Catalog.id == catalog_id).first()
    if not catalog:
        raise HTTPException(status_code=404, detail="Catálogo no encontrado")

    # 2) Buscar parts del catálogo
    parts = db.query(models.Part).filter(models.Part.catalog_id == catalog_id).all()
    part_ids = [p.id for p in parts]

    # 3) Borrar dependencias (si NO tienes cascade configurado)
    if part_ids:
        db.query(models.PriceTier).filter(models.PriceTier.part_id.in_(part_ids)).delete(synchronize_session=False)
        db.query(models.PartAttribute).filter(models.PartAttribute.part_id.in_(part_ids)).delete(synchronize_session=False)
        db.query(models.Part).filter(models.Part.id.in_(part_ids)).delete(synchronize_session=False)

    # 4) Borrar el catálogo
    db.delete(catalog)
    db.commit()

    return {"message": "Catálogo eliminado", "catalog_id": catalog_id, "deleted_parts": len(part_ids)}


@app.get("/catalogs", response_model=List[schemas.CatalogListOut])
def list_catalogs(db: Session = Depends(get_db)):
    """
    Devuelve la lista de catálogos con el nombre de su proveedor.
    """
    catalogs = (
        db.query(models.Catalog)
        .join(models.Supplier)
        .order_by(models.Catalog.created_at.desc())
        .all()
    )
    
    # Transformamos los datos para que coincidan con el esquema
    results = []
    for cat in catalogs:
        results.append({
            "id": cat.id,
            "supplier_name": cat.supplier.name,
            "year": cat.year,
            "original_filename": cat.original_filename,
            "created_at": cat.created_at
        })
    
    return results
from fastapi.responses import StreamingResponse
from collections import defaultdict

# ============================================================
#  Endpoint: Exportación (Part Number | Empresa | Precios)
#  - NO usa joinedload(models.Part.supplier) para evitar 500
#  - Trae todos los tramos en una sola columna
# ============================================================
@app.get("/parts/search/export")
def export_search_to_excel(
    query: str = Query(..., alias="q", min_length=1),
    supplier: str = Query(None, description="Nombre proveedor opcional (ej: STUKERJURGEN)"),
    currency: str = Query(None, description="Moneda opcional (ej: USD / EUR)"),
    db: Session = Depends(get_db),
):
    raw = (query or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="q vacío")

    terms = [t.strip() for t in re.split(r"[,\n;\t]+", raw) if t.strip()]
    terms = terms[:30]

    groups = []
    for t in terms:
        t_norm = normalize_pn(t)

        like_prefix = f"{t}%"
        like_any = f"%{t}%"
        like_prefix_norm = f"{t_norm}%"
        like_any_norm = f"%{t_norm}%"

        groups.append(
            or_(
                # normal
                models.Part.part_number_full.ilike(like_prefix),
                models.Part.part_number_root.ilike(like_prefix),
                models.Part.description.ilike(like_any),
                models.PartAttribute.attr_value.ilike(like_any),
                models.PartAlias.code.ilike(like_any),

                # normalizado (sin guiones/espacios)
                sql_normalize(models.Part.part_number_full, db).ilike(like_prefix_norm),
                sql_normalize(models.Part.part_number_root, db).ilike(like_prefix_norm),
                sql_normalize(models.PartAlias.code, db).ilike(like_any_norm),
            )
        )

    combined_filter = or_(*groups)

    q = (
        db.query(models.Part, models.Supplier)
        .join(models.Supplier, models.Part.supplier_id == models.Supplier.id)
        .outerjoin(models.PartAttribute, models.PartAttribute.part_id == models.Part.id)
        .outerjoin(models.PartAlias, models.PartAlias.part_id == models.Part.id)
        .filter(combined_filter)
    )

    # filtros opcionales
    if supplier:
        q = q.filter(models.Supplier.name.ilike(f"%{supplier}%"))

    if currency:
        q = q.filter(models.Part.currency == currency.upper())

    rows = q.order_by(models.Part.part_number_full).limit(500).all()

    # --- traer tiers en 1 query (sin depender de relación ORM) ---
    part_ids = [p.id for (p, s) in rows]
    tiers_by_part = defaultdict(list)

    if part_ids:
        tier_rows = (
            db.query(models.PriceTier)
            .filter(models.PriceTier.part_id.in_(part_ids))
            .order_by(models.PriceTier.part_id, models.PriceTier.min_qty)
            .all()
        )
        for tr in tier_rows:
            tiers_by_part[tr.part_id].append(tr)

    def format_prices(part) -> str:
        tiers = tiers_by_part.get(part.id, []) or []

        if tiers:
            chunks = []
            for t in tiers:
                cur = t.currency or part.currency or ""
                if t.max_qty is not None:
                    label = f"{t.min_qty}-{t.max_qty}"
                else:
                    label = f">={t.min_qty}"
                chunks.append(f"{label}: {t.unit_price} {cur}".strip())
            return " | ".join(chunks)

        if part.base_price is not None:
            cur = part.currency or ""
            minq = part.min_qty_default or 1
            return f">={minq}: {part.base_price} {cur}".strip()

        return ""

    export_rows = []
    for part, sup in rows:
        export_rows.append(
            {
                "Part Number": part.part_number_full or part.part_number_root or "",
                "Empresa": sup.name if sup else "",
                "Precios": format_prices(part),
            }
        )

    df = pd.DataFrame(export_rows, columns=["Part Number", "Empresa", "Precios"])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Resultados", index=False)
        ws = writer.sheets["Resultados"]
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 22
        ws.column_dimensions["C"].width = 70

    output.seek(0)
    filename = f"busqueda_{normalize_pn(raw)[:20] or 'resultados'}.xlsx"

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
