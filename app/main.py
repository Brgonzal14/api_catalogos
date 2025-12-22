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
from sqlalchemy.orm import Session, joinedload
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError
from sqlalchemy import or_
from typing import List, Dict, Tuple, Optional
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

from .db import Base, engine, get_db
from . import models, schemas
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


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
            # Caso STUKERJURGEN Hansair (y otros PDFs tabulares)
            dfs = [read_pdf_tables(content)]

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

        # Columnas de precios por tramo (p.ej. "qty/ea 25-99", ...)
        tier_specs: List[Dict[str, Optional[int]]] = []
        for col in df.columns:
            col_lower = str(col).strip().lower()
            if col_lower in qty_ranges_by_col_lower:
                range_text = qty_ranges_by_col_lower[col_lower]
                min_q, max_q = parse_qty_range(range_text)
                tier_specs.append(
                    {"col": col, "min_qty": min_q, "max_qty": max_q}
                )

        # ===============================
        # 4) RECORRER FILAS E INSERTAR EN BD
        # ===============================
        last_part_code: Optional[str] = None  # para reutilizar ArticleNo en filas sin código

        for _, row in df.iterrows():
            # ---------- CÓDIGO DE PARTE ----------
            candidates = [
                row.get("part_number"),
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
# ============================================================
@app.get("/parts/search")
def search_parts(
    query: str = Query(
        ...,
        alias="q",  # ?q= en la URL
        min_length=1,
        description="Código o parte de la descripción",
    ),
    db: Session = Depends(get_db),
):
    q = query.strip()
    if not q:
        return []

    like_prefix = f"{q}%"
    like_any = f"%{q}%"

    # JOIN con Supplier y OUTER JOIN con PartAttribute
    rows = (
        db.query(models.Part, models.Supplier)
        .join(models.Supplier, models.Part.supplier_id == models.Supplier.id)
        .outerjoin(
            models.PartAttribute,
            models.PartAttribute.part_id == models.Part.id,
        )
        .filter(
            or_(
                # códigos principales
                models.Part.part_number_full.ilike(like_prefix),
                models.Part.part_number_root.ilike(like_prefix),
                # descripción
                models.Part.description.ilike(like_any),
                # 🔹 ahora también busca en TODOS los atributos
                models.PartAttribute.attr_value.ilike(like_any),
            )
        )
        .order_by(models.Part.part_number_full)
        .limit(50)
        .all()
    )

    results = []
    seen_parts = set()  # para evitar duplicados por el JOIN con atributos

    for part, supplier in rows:
        if part.id in seen_parts:
            continue
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

