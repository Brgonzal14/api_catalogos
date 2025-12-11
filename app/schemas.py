from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime 

class PriceTierBase(BaseModel):
    min_qty: int
    max_qty: Optional[int] = None
    unit_price: float
    currency: str

class PriceTierOut(PriceTierBase):
    id: int
    class Config:
        from_attributes = True

class PartAttributeOut(BaseModel):
    attr_name: str
    attr_value: str
    class Config:
        from_attributes = True

class PartOut(BaseModel):
    id: int
    part_number_full: str
    part_number_root: str
    description: str | None = None
    currency: str | None = None
    base_price: float | None = None
    price_tiers: List[PriceTierOut] = []
    attributes: List[PartAttributeOut] = []

    class Config:
        from_attributes = True


class PartSearchOut(BaseModel):
    id: int
    part_number_full: str
    description: str
    supplier_name: str
    currency: str
    base_price: float
    catalogs: List[str]  # nombres de los catálogos donde aparece

    class Config:
        orm_mode = True


class CatalogListOut(BaseModel):
    id: int
    supplier_name: str
    year: int
    original_filename: str
    created_at: datetime

    class Config:
        from_attributes = True