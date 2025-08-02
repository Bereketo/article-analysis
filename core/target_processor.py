import os
import json
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

@dataclass
class TargetEntity:
    """Enhanced target entity with comprehensive metadata"""
    name: str
    entity_type: Literal["company", "individual"]
    known_aliases: List[str] = field(default_factory=list)
    geographic_scope: List[str] = field(default_factory=lambda: ["global"])
    time_frame_years: int = 5
    industry_sector: Optional[str] = None
    regulatory_jurisdictions: List[str] = field(default_factory=list)
    kyc_identifiers: Dict[str, str] = field(default_factory=dict)  # CIN, PAN, etc.
    
    def get_search_variations(self) -> List[str]:
        """Generate all possible search variations"""
        variations = [self.name] + self.known_aliases
        
        # Add common variations
        for name in [self.name] + self.known_aliases:
            variations.extend([
                f'"{name}"',  # Exact match
                name.replace(" ", ""),  # No spaces
                name.replace(".", ""),  # No dots
                name.replace(",", ""),  # No commas
            ])
        
        return list(set(variations))

class TargetProcessor:
    """Enhanced target processing with multi-source validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def process_target_input(self, target_name: str, entity_type: str = "company") -> TargetEntity:
        """Process initial target input with validation"""
        
        # Clean and normalize target name
        cleaned_name = self._clean_target_name(target_name)
        
        # Create enhanced target entity
        target = TargetEntity(
            name=cleaned_name,
            entity_type=entity_type,
            time_frame_years=5
        )
        
        # Auto-detect additional metadata
        target = self._enrich_target_metadata(target)
        
        return target
    
    def _clean_target_name(self, name: str) -> str:
        """Clean and normalize target name"""
        # Remove extra spaces, special characters
        cleaned = " ".join(name.strip().split())
        
        # Handle common corporate suffixes
        corporate_suffixes = ["Ltd", "Limited", "Inc", "Corp", "LLC", "Pvt"]
        for suffix in corporate_suffixes:
            if cleaned.endswith(f" {suffix}."):
                cleaned = cleaned.replace(f" {suffix}.", f" {suffix}")
        
        return cleaned
    
    def _enrich_target_metadata(self, target: TargetEntity) -> TargetEntity:
        """Enrich target with additional metadata"""
        
        # Auto-detect industry and jurisdiction based on name patterns
        if target.entity_type == "company":
            target.industry_sector = self._detect_industry(target.name)
            target.regulatory_jurisdictions = self._detect_jurisdictions(target.name)
        
        return target
    
    def _detect_industry(self, name: str) -> Optional[str]:
        """Auto-detect industry sector"""
        industry_keywords = {
            "financial": ["bank", "finance", "capital", "investment", "fund"],
            "technology": ["tech", "software", "systems", "digital", "cyber"],
            "pharmaceutical": ["pharma", "bio", "medical", "health"],
            "energy": ["oil", "gas", "energy", "power", "solar"],
            "manufacturing": ["manufacturing", "industrial", "steel", "auto"]
        }
        
        name_lower = name.lower()
        for industry, keywords in industry_keywords.items():
            if any(keyword in name_lower for keyword in keywords):
                return industry
        
        return None
    
    def _detect_jurisdictions(self, name: str) -> List[str]:
        """Auto-detect regulatory jurisdictions"""
        jurisdiction_indicators = {
            "india": ["pvt", "ltd", "limited"],
            "usa": ["inc", "corp", "llc"],
            "uk": ["plc", "limited"],
            "singapore": ["pte"]
        }
        
        name_lower = name.lower()
        jurisdictions = []
        
        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in name_lower for indicator in indicators):
                jurisdictions.append(jurisdiction)
        
        return jurisdictions if jurisdictions else ["global"]