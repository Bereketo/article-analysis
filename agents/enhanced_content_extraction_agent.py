import asyncio
from typing import Dict, List, Any, Optional
import logging
from core.target_processor import TargetProcessor, TargetEntity
from sources.official_sources_manager import OfficialSourcesManager
from analysis.corporate_structure_analyzer import CorporateStructureAnalyzer
from agents.improved_content_extraction_agent import ImprovedContentExtractionAgent

class EnhancedContentExtractionAgent(ImprovedContentExtractionAgent):
    """Enhanced agent with comprehensive target processing and multi-source analysis"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.target_processor = TargetProcessor()
        self.official_sources_manager = OfficialSourcesManager()
        self.corporate_analyzer = CorporateStructureAnalyzer()
        
        self.logger.info("Enhanced Content Extraction Agent initialized with multi-source capabilities")
    
    async def comprehensive_adverse_screening(
        self, 
        target_name: str, 
        entity_type: str = "company",
        include_corporate_structure: bool = True
    ) -> Dict[str, Any]:
        """Comprehensive adverse screening with all optimizations"""
        
        self.logger.info(f"🎯 Starting comprehensive screening for: {target_name}")
        
        # Step 1: Enhanced target processing
        target = self.target_processor.process_target_input(target_name, entity_type)
        self.logger.info(f"📋 Target processed: {target.name} ({target.entity_type})")
        
        # Step 2: Corporate structure analysis (for companies)
        corporate_structure = None
        if include_corporate_structure and target.entity_type == "company":
            self.logger.info("🏢 Analyzing corporate structure...")
            corporate_structure = await self.corporate_analyzer.analyze_corporate_structure(target)
            
            # Extend search terms with corporate entities
            extended_terms = self.corporate_analyzer.generate_extended_search_terms(corporate_structure)
            self.logger.info(f"📈 Extended search terms: {len(extended_terms)} entities identified")
        
        # Step 3: Official sources search
        self.logger.info("🏛️ Searching official sources...")
        official_results = await self.official_sources_manager.search_official_sources(
            target.name, 
            target.regulatory_jurisdictions
        )
        
        # Step 4: Enhanced alias detection
        self.logger.info("🔍 Enhanced alias detection...")
        alias_results = await self._enhanced_alias_detection(target, corporate_structure)
        
        # Step 5: Comprehensive adverse search
        self.logger.info("⚠️ Comprehensive adverse search...")
        all_search_terms = self._generate_comprehensive_search_terms(target, alias_results, corporate_structure)
        adverse_results = await self._execute_comprehensive_adverse_search(all_search_terms)
        
        # Step 6: Compile comprehensive results
        comprehensive_results = {
            "target_info": {
                "name": target.name,
                "entity_type": target.entity_type,
                "geographic_scope": target.geographic_scope,
                "time_frame_years": target.time_frame_years,
                "industry_sector": target.industry_sector,
                "regulatory_jurisdictions": target.regulatory_jurisdictions
            },
            "alias_analysis": alias_results,
            "corporate_structure": corporate_structure.__dict__ if corporate_structure else None,
            "official_sources_results": official_results,
            "adverse_screening_results": adverse_results,
            "summary": self._generate_comprehensive_summary(adverse_results),
            "risk_assessment": self._generate_risk_assessment(adverse_results, official_results)
        }
        
        self.logger.info("✅ Comprehensive screening completed")
        return comprehensive_results
    
    async def _enhanced_alias_detection(self, target: TargetEntity, corporate_structure) -> Dict[str, Any]:
        """Enhanced alias detection with corporate structure integration"""
        
        # Use existing alias agent with enhancements
        base_aliases = target.get_search_variations()
        
        # Add corporate structure aliases
        if corporate_structure:
            for entity_list in [
                corporate_structure.subsidiaries,
                corporate_structure.joint_ventures,
                corporate_structure.historical_entities
            ]:
                base_aliases.extend([entity.name for entity in entity_list])
        
        return {
            "primary_name": target.name,
            "all_aliases": list(set(base_aliases)),
            "corporate_aliases": [entity.name for entity in corporate_structure.subsidiaries] if corporate_structure else [],
            "historical_names": [entity.name for entity in corporate_structure.historical_entities] if corporate_structure else []
        }
    
    def _generate_comprehensive_search_terms(
        self, 
        target: TargetEntity, 
        alias_results: Dict[str, Any], 
        corporate_structure
    ) -> List[str]:
        """Generate comprehensive search terms for adverse screening"""
        
        # Base adverse keywords from your original requirements
        adverse_keywords = [
            "fraud", "default", "kickback", "scandal", "probe", "penalty", "misconduct",
            "CBI", "imprisonment", "police", "scam", "vigilance", "litigation", "arrests",
            "accused", "accuse", "alleged", "allegedly", "sentenced", "illegal", "lawsuit",
            "ban", "banned", "politics", "political", "murder", "bribe", "bribed",
            "rape", "raped", "raping", "robbed", "rob", "theft", "charged", "mobbed",
            "jail", "jailed", "victim", "underworld", "terrorists", "terrorist", "terrorism",
            "investigate", "investigated", "investigation", "penalized", "probed", "case",
            "criminal", "crime", "abuse", "abused", "strike", "accident", "manipulated",
            "manipulation", "downfall", "loss", "cronies", "absconds", "absconding",
            "absconded", "lobbying", "compliance", "complying", "prison", "imprisoned",
            "warned", "warn", "warning", "terminate"
        ]
        
        search_terms = []
        all_names = alias_results["all_aliases"]
        
        # Generate comprehensive search combinations
        for name in all_names[:10]:  # Limit to prevent too many queries
            # Exact name search first
            search_terms.append(f'"{name}"')
            
            # Adverse keyword combinations
            for keyword in adverse_keywords[:20]:  # Limit keywords
                search_terms.append(f'"{name}" AND {keyword}')
        
        return search_terms
    
    async def _execute_comprehensive_adverse_search(self, search_terms: List[str]) -> Dict[str, Any]:
        """Execute comprehensive adverse search with all terms"""
        
        # Use your existing search capabilities
        all_results = []
        
        # Process in batches to avoid overwhelming APIs
        batch_size = 10
        for i in range(0, len(search_terms), batch_size):
            batch = search_terms[i:i + batch_size]
            
            batch_tasks = []
            for term in batch:
                # Use your existing search methods
                task = self._search_single_term(term)
                batch_tasks.append(task)
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            all_results.extend([r for r in batch_results if not isinstance(r, Exception)])
            
            # Rate limiting
            await asyncio.sleep(1)
        
        return {
            "total_searches": len(search_terms),
            "successful_searches": len(all_results),
            "results": all_results
        }
    
    async def _search_single_term(self, search_term: str) -> Dict[str, Any]:
        """Search single term using existing capabilities"""
        # This would integrate with your existing search methods
        # Placeholder implementation
        return {
            "search_term": search_term,
            "results": [],
            "timestamp": "2024-01-01T00:00:00Z"
        }
    
    def _generate_comprehensive_summary(self, adverse_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of findings"""
        return {
            "total_adverse_mentions": len(adverse_results.get("results", [])),
            "high_risk_findings": 0,  # Would be calculated based on actual results
            "medium_risk_findings": 0,
            "low_risk_findings": 0,
            "categories_found": [],
            "recommendation": "Further investigation required"
        }
    
    def _generate_risk_assessment(self, adverse_results: Dict[str, Any], official_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""
        return {
            "overall_risk_level": "Medium",  # Would be calculated
            "confidence_score": 0.75,
            "key_risk_factors": [],
            "mitigation_recommendations": [],
            "next_steps": [
                "Review official source findings",
                "Analyze corporate structure risks",
                "Monitor for ongoing developments"
            ]
        }

# Usage example
async def main():
    agent = EnhancedContentExtractionAgent()
    
    result = await agent.comprehensive_adverse_screening(
        target_name="Infosys Limited",
        entity_type="company",
        include_corporate_structure=True
    )
    
    print("Comprehensive screening completed!")
    print(f"Found {len(result['alias_analysis']['all_aliases'])} aliases")
    print(f"Analyzed {len(result['official_sources_results'])} official sources")

if __name__ == "__main__":
    asyncio.run(main())