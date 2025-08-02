import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import json
import re
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_openai import AzureChatOpenAI

@dataclass
class CorporateEntity:
    name: str
    relationship_type: str  # "subsidiary", "parent", "joint_venture", "affiliate", "historical"
    ownership_percentage: Optional[float] = None
    jurisdiction: Optional[str] = None
    status: str = "active"  # "active", "dissolved", "merged"
    confidence_score: float = 0.0
    source_urls: List[str] = None
    
    def __post_init__(self):
        if self.source_urls is None:
            self.source_urls = []

@dataclass
class CorporateStructure:
    target_entity: str
    parent_companies: List[CorporateEntity]
    subsidiaries: List[CorporateEntity]
    joint_ventures: List[CorporateEntity]
    affiliates: List[CorporateEntity]
    historical_entities: List[CorporateEntity]

class CorporateStructureAnalyzer:
    """Comprehensive corporate structure analysis with real search integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_search_tools()
        self._setup_llm()
        
    def _setup_search_tools(self):
        """Setup search tools for corporate structure analysis"""
        bing_wrapper = BingSearchAPIWrapper(
            bing_subscription_key="822e2402879b4c78b00434c7f0f4c201",
            search_kwargs={"mkt": "en-IN"}
        )
        self.bing_tool = BingSearchResults(api_wrapper=bing_wrapper, num_results=15)
        
    def _setup_llm(self):
        """Setup LLM for entity extraction"""
        self.llm = AzureChatOpenAI(
            openai_api_key="1f5d1bb6920844248ea17f61f73f82ac",
            azure_endpoint="https://ai-gpt-echo.openai.azure.com",
            azure_deployment="o3-mini",
            openai_api_version="2024-12-01-preview",
            temperature=0.1,
            max_completion_tokens=2000  # Use max_completion_tokens for o3-mini
        )
    
    async def analyze_corporate_structure(self, target: TargetEntity) -> CorporateStructure:
        """Comprehensive corporate structure analysis"""
        
        if target.entity_type != "company":
            return CorporateStructure(
                target_entity=target.name,
                parent_companies=[],
                subsidiaries=[],
                joint_ventures=[],
                affiliates=[],
                historical_entities=[]
            )
        
        # Parallel analysis of different relationship types
        tasks = [
            self._find_parent_companies(target),
            self._find_subsidiaries(target),
            self._find_joint_ventures(target),
            self._find_affiliates(target),
            self._find_historical_entities(target)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return CorporateStructure(
            target_entity=target.name,
            parent_companies=results[0] if not isinstance(results[0], Exception) else [],
            subsidiaries=results[1] if not isinstance(results[1], Exception) else [],
            joint_ventures=results[2] if not isinstance(results[2], Exception) else [],
            affiliates=results[3] if not isinstance(results[3], Exception) else [],
            historical_entities=results[4] if not isinstance(results[4], Exception) else []
        )
    
    async def _find_parent_companies(self, target: TargetEntity) -> List[CorporateEntity]:
        """Find parent companies and holding structures"""
        
        search_queries = [
            f"{target.name} parent company",
            f"{target.name} holding company",
            f"{target.name} owned by",
            f"{target.name} subsidiary of",
            f'"{target.name}" parent',
            f"{target.name} ultimate beneficial owner"
        ]
        
        # This would integrate with your search system
        parent_entities = []
        
        # Placeholder for actual search implementation
        for query in search_queries:
            # Search and parse results
            # Extract parent company information
            pass
        
        return parent_entities
    
    async def _find_subsidiaries(self, target: TargetEntity) -> List[CorporateEntity]:
        """Find subsidiaries and controlled entities"""
        
        search_queries = [
            f"{target.name} subsidiaries",
            f"{target.name} owns",
            f"{target.name} controlled entities",
            f'"{target.name}" subsidiary',
            f"{target.name} group companies",
            f"{target.name} wholly owned"
        ]
        
        subsidiaries = []
        
        # Enhanced search for subsidiaries
        for query in search_queries:
            # Implementation would use your existing search tools
            pass
        
        return subsidiaries
    
    async def _find_joint_ventures(self, target: TargetEntity) -> List[CorporateEntity]:
        """Find joint ventures and partnerships"""
        
        search_queries = [
            f"{target.name} joint venture",
            f"{target.name} partnership",
            f"{target.name} JV",
            f'"{target.name}" joint venture',
            f"{target.name} strategic partnership",
            f"{target.name} consortium"
        ]
        
        joint_ventures = []
        
        for query in search_queries:
            # Implementation would use your existing search tools
            pass
        
        return joint_ventures
    
    async def _find_affiliates(self, target: TargetEntity) -> List[CorporateEntity]:
        """Find affiliated companies and related entities"""
        
        search_queries = [
            f"{target.name} affiliate",
            f"{target.name} related company",
            f"{target.name} associated company",
            f'"{target.name}" affiliate',
            f"{target.name} group entity"
        ]
        
        affiliates = []
        
        for query in search_queries:
            # Implementation would use your existing search tools
            pass
        
        return affiliates
    
    async def _find_historical_entities(self, target: TargetEntity) -> List[CorporateEntity]:
        """Find historical entities (mergers, acquisitions, name changes)"""
        
        search_queries = [
            f"{target.name} formerly known as",
            f"{target.name} previous name",
            f"{target.name} merger",
            f"{target.name} acquisition",
            f'"{target.name}" formerly',
            f"{target.name} renamed from",
            f"{target.name} corporate restructuring"
        ]
        
        historical_entities = []
        
        for query in search_queries:
            # Implementation would use your existing search tools
            pass
        
        return historical_entities
    
    def generate_extended_search_terms(self, structure: CorporateStructure) -> List[str]:
        """Generate extended search terms from corporate structure"""
        
        search_terms = [structure.target_entity]
        
        # Add all related entities
        for entity_list in [
            structure.parent_companies,
            structure.subsidiaries,
            structure.joint_ventures,
            structure.affiliates,
            structure.historical_entities
        ]:
            search_terms.extend([entity.name for entity in entity_list])
        
        # Generate combination searches
        combination_terms = []
        for entity in structure.subsidiaries[:5]:  # Limit to top 5
            combination_terms.append(f"{structure.target_entity} {entity.name}")
        
        search_terms.extend(combination_terms)
        
        return list(set(search_terms))
    
    async def _find_subsidiaries(self, target) -> List[CorporateEntity]:
        """Find subsidiaries using real search integration"""
        
        search_queries = [
            f"{target.name} subsidiaries list",
            f"{target.name} owns companies",
            f"{target.name} controlled entities",
            f'"{target.name}" subsidiary companies',
            f"{target.name} group companies structure",
            f"{target.name} wholly owned subsidiaries"
        ]
        
        subsidiaries = []
        
        for query in search_queries:
            try:
                # Execute search
                search_results = self.bing_tool.run(query)
                
                # Extract entities using LLM
                extracted_entities = await self._extract_corporate_entities(
                    search_results, 
                    target.name, 
                    "subsidiary"
                )
                
                subsidiaries.extend(extracted_entities)
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error searching subsidiaries with query '{query}': {e}")
        
        # Deduplicate and return
        return self._deduplicate_entities(subsidiaries)
    
    async def _find_parent_companies(self, target) -> List[CorporateEntity]:
        """Find parent companies using real search integration"""
        
        search_queries = [
            f"{target.name} parent company",
            f"{target.name} holding company",
            f"{target.name} owned by company",
            f"{target.name} subsidiary of",
            f'"{target.name}" parent organization',
            f"{target.name} ultimate beneficial owner"
        ]
        
        parent_companies = []
        
        for query in search_queries:
            try:
                search_results = self.bing_tool.run(query)
                
                extracted_entities = await self._extract_corporate_entities(
                    search_results, 
                    target.name, 
                    "parent"
                )
                
                parent_companies.extend(extracted_entities)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error searching parent companies with query '{query}': {e}")
        
        return self._deduplicate_entities(parent_companies)
    
    async def _find_joint_ventures(self, target) -> List[CorporateEntity]:
        """Find joint ventures using real search integration"""
        
        search_queries = [
            f"{target.name} joint venture partners",
            f"{target.name} JV companies",
            f"{target.name} strategic partnerships",
            f'"{target.name}" joint venture',
            f"{target.name} consortium members",
            f"{target.name} partnership agreements"
        ]
        
        joint_ventures = []
        
        for query in search_queries:
            try:
                search_results = self.bing_tool.run(query)
                
                extracted_entities = await self._extract_corporate_entities(
                    search_results, 
                    target.name, 
                    "joint_venture"
                )
                
                joint_ventures.extend(extracted_entities)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Error searching joint ventures with query '{query}': {e}")
        
        return self._deduplicate_entities(joint_ventures)
    
    async def _extract_corporate_entities(
        self, 
        search_results: str, 
        target_name: str, 
        relationship_type: str
    ) -> List[CorporateEntity]:
        """Extract corporate entities from search results using LLM"""
        
        extraction_prompt = f"""
        Analyze the following search results and extract corporate entities that have a {relationship_type} relationship with "{target_name}".

        Search Results:
        {search_results}

        Extract entities in this JSON format:
        {{
            "entities": [
                {{
                    "name": "Company Name",
                    "relationship_type": "{relationship_type}",
                    "ownership_percentage": 75.5,
                    "jurisdiction": "India",
                    "status": "active",
                    "confidence_score": 0.85,
                    "source_evidence": "Brief evidence from search results"
                }}
            ]
        }}

        Rules:
        1. Only extract entities with clear {relationship_type} relationship to {target_name}
        2. Confidence score 0.0-1.0 based on evidence strength
        3. Include ownership percentage if mentioned
        4. Status: "active", "dissolved", "merged", or "unknown"
        5. Jurisdiction if mentioned
        6. Minimum confidence 0.6 to include
        """
        
        try:
            response = await self.llm.ainvoke(extraction_prompt)
            
            # Parse JSON response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response.content[json_start:json_end]
                parsed_data = json.loads(json_str)
                
                entities = []
                for entity_data in parsed_data.get("entities", []):
                    if entity_data.get("confidence_score", 0) >= 0.6:
                        entity = CorporateEntity(
                            name=entity_data["name"],
                            relationship_type=relationship_type,
                            ownership_percentage=entity_data.get("ownership_percentage"),
                            jurisdiction=entity_data.get("jurisdiction"),
                            status=entity_data.get("status", "unknown"),
                            confidence_score=entity_data.get("confidence_score", 0.0),
                            source_urls=[]
                        )
                        entities.append(entity)
                
                return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
        
        return []
    
    def _deduplicate_entities(self, entities: List[CorporateEntity]) -> List[CorporateEntity]:
        """Remove duplicate entities and keep highest confidence"""
        
        entity_map = {}
        
        for entity in entities:
            # Normalize name for comparison
            normalized_name = entity.name.lower().strip()
            
            if normalized_name not in entity_map:
                entity_map[normalized_name] = entity
            else:
                # Keep entity with higher confidence
                if entity.confidence_score > entity_map[normalized_name].confidence_score:
                    entity_map[normalized_name] = entity
        
        return list(entity_map.values())
