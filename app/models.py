from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, Float, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .db import Base

class Supplier(Base):
    __tablename__ = "suppliers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    catalogs = relationship("Catalog", back_populates="supplier")


class Catalog(Base):
    __tablename__ = "catalogs"

    id = Column(Integer, primary_key=True, index=True)
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    year = Column(Integer, index=True)
    original_filename = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    supplier = relationship("Supplier", back_populates="catalogs")
    parts = relationship("Part", back_populates="catalog")


class Part(Base):
    __tablename__ = "parts"

    id = Column(Integer, primary_key=True, index=True)
    catalog_id = Column(Integer, ForeignKey("catalogs.id"))
    supplier_id = Column(Integer, ForeignKey("suppliers.id"))
    part_number_full = Column(String, index=True)
    part_number_root = Column(String, index=True)
    description = Column(String)
    currency = Column(String(3))
    base_price = Column(Float)
    min_qty_default = Column(Integer, default=1)
    is_active = Column(Boolean, default=True)

    catalog = relationship("Catalog", back_populates="parts")
    price_tiers = relationship("PriceTier", back_populates="part")
    attributes = relationship("PartAttribute", back_populates="part")
    aliases = relationship(
        "PartAlias",
        back_populates="part",
        cascade="all, delete-orphan",
    )

class PriceTier(Base):
    __tablename__ = "price_tiers"

    id = Column(Integer, primary_key=True, index=True)
    part_id = Column(Integer, ForeignKey("parts.id"))
    min_qty = Column(Integer)
    max_qty = Column(Integer, nullable=True)
    unit_price = Column(Float)
    currency = Column(String(3))

    part = relationship("Part", back_populates="price_tiers")


class PartAttribute(Base):
    __tablename__ = "part_attributes"

    id = Column(Integer, primary_key=True, index=True)
    part_id = Column(Integer, ForeignKey("parts.id"))
    attr_name = Column(String)
    attr_value = Column(String)

    part = relationship("Part", back_populates="attributes")

class PartAlias(Base):
    __tablename__ = "part_aliases"

    id = Column(Integer, primary_key=True, index=True)
    part_id = Column(Integer, ForeignKey("parts.id"), index=True, nullable=False)
    code = Column(String, index=True, nullable=False)
    source = Column(String)  # p.ej. "HOLMCO_END_UNIT"

    part = relationship("Part", back_populates="aliases")