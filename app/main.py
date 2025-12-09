from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    Depends,
    Query,
    HTTPException,
)
from sqlalchemy.orm import Session
from typing import List, Dict, Tuple, Optional
import os
import io
import time
import zipfile
import xml.etree.ElementTree as ET
import re
import asyncio

import pandas as pd
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError
from sqlalchemy import or_

from .db import Base, engine, get_db
from . import models, schemas


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


app = FastAPI(title="API Catálogos Aeronáuticos", lifespan=lifespan)


# ============================================================
#  Helpers de BD
# ============================================================
def get_or_create_supplier(db: Session, name: str) -> models.Supplier:
    supplier = db.query(models.Supplier).filter(models.Supplier.name == name).first()
    if supplier:
        return supplier
    supplier = models.Supplier(name=name)
    db.add(supplier)
    db.commit()
    db.refresh(supplier)
    return supplier


def create_catalog(db: Session, supplier: models.Supplier, year: int, filename: str) -> models.Catalog:
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
    if code is None:
        return "", ""
    code = code.strip()
    root = code.split("-")[0]
    return code, root


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

    Requiere instalar pdfplumber:
        pip install pdfplumber
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
    Detecta:
    - fila de encabezados (buscando 'part number', 'description' o 'item ...')
    - fila inmediatamente debajo con los rangos '25-99', '100-249', etc.
      y construye nombres de columnas únicos tipo:
      'Qty/ea 25-99', 'Qty/ea 100-249', ...

    Devuelve:
      - df recortado solo a las filas de datos
      - dict {nombre_columna_lowercase: texto_rango}
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

    if header_row_idx is not None:
        header_row = df.iloc[header_row_idx]
        next_row = df.iloc[header_row_idx + 1] if header_row_idx + 1 < len(df) else None

        col_names: List[str] = []
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
            "part number": "part_number",
            "part_number": "part_number",
            "part numbep": "part_number",
            "pn": "part_number",
            "p/n": "part_number",
            "part-no.": "part_number",
            "part-no": "part_number",
            "part no.": "part_number",

            # Article / ArticleNo
            "articleno.": "article_no",
            "article no.": "article_no",
            "article no": "article_no",

            # columnas ITEM (AIRTEC-BRAIDS)
            "item (standard)": "part_number",
            "item (other)": "part_number",
            "item": "part_number",

            # descripción
            "descripcion": "description",
            "description": "description",

            # precio
            "price": "price",
            "precio": "price",
            "master $": "price",  # catálogos tipo Master $

            # moneda
            "currency": "currency",
            "moneda": "currency",

            # cantidades mínimas / máximas
            "min qty": "min_qty",
            "min_qty": "min_qty",
            "from qty": "min_qty",
            "to qty": "max_qty",
            "max qty": "max_qty",
            "quantity": "min_qty",
            "qty": "min_qty",

            # otros posibles
            "unit code": "unit_code",
            "unit": "unit_code",
            "lead time": "lead_time",
            "lifecycle": "lifecycle",
            "cumulative": "cumulative",
        }

        normalized_cols: Dict[str, str] = {}
        for col in df.columns:
            raw_name = str(col).strip()
            col_key = raw_name.lower()

            # 1) Primero, si hay mapeo exacto en column_map, usamos ese
            if col_key in column_map:
                mapped = column_map[col_key]

            # 2) PANASONIC Master Price List: columnas tipo "Master S2 2023", "Master S2 2024", etc.
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
        #     ArticleNo = código principal, Part-No. = código del fabricante
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

            # Si esta fila no trae código (ArticleNo en blanco), usamos el último
            if (raw_code is None or str(raw_code).strip() == "") and last_part_code is not None:
                raw_code = last_part_code

            # Si sigue sin código, no podemos usar esta fila
            if raw_code is None or str(raw_code).strip() == "":
                continue

            part_number_full, part_number_root = normalize_part_number(str(raw_code))

            # Si el "código" no tiene ningún dígito (p.ej. 'ArticleNo.' de la cabecera),
            # lo interpretamos como fila de encabezado y la ignoramos.
            if not any(ch.isdigit() for ch in part_number_root):
                continue

            # Guardamos el último código visto para las siguientes filas sin ArticleNo
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
                # buscamos el primer número entero que aparezca en el texto
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
                    "part_number",
                    "article_no",
                    "description",
                    "price",
                    "currency",
                    "min_qty",
                    "max_qty",
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
#  Endpoint: búsqueda de piezas
# ============================================================
@app.get("/parts/search", response_model=List[schemas.PartOut])
def search_parts(
    query: str = Query(..., min_length=1, description="Código o parte de la descripción"),
    db: Session = Depends(get_db),
):
    q = query.strip()
    if not q:
        return []

    like_prefix = f"{q}%"    # empieza con lo que escribes
    like_any = f"%{q}%"      # contiene lo que escribes

    parts = (
        db.query(models.Part)
        .filter(
            or_(
                models.Part.part_number_full.ilike(like_prefix),
                models.Part.part_number_root.ilike(like_prefix),
                models.Part.description.ilike(like_any),
            )
        )
        .order_by(models.Part.part_number_full)
        .limit(50)
        .all()
    )

    return parts
