import os
import json
import re
import logging
from typing import Dict, List, Any
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)

class AliasGenerationImproved:
    """Improved alias generation agent using direct LLM queries"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.3,
            max_tokens=2000
        )
    
    async def generate_aliases(self, company_name: str, country: str = "India") -> Dict[str, Any]:
        """Generate comprehensive aliases using direct LLM query"""
        
        prompt = f"""You are a corporate research expert specializing in company name variations and historical information.

TASK: Research and provide comprehensive information about "{company_name}" in {country}.

INSTRUCTIONS:
1. Identify the current official company name
2. Find ALL former names, previous names, historical names
3. Include common variations, abbreviations, and trading names
4. Find stock ticker symbols (NSE, BSE, NYSE, NASDAQ, etc.)
5. Identify parent company or holding company
6. Focus on finding accurate company information (adverse queries will be generated automatically)

SEARCH FOR THESE PATTERNS:
- "formerly known as"
- "previously called" 
- "renamed from"
- "was known as"
- "earlier called"
- "changed name from"
- "rebranded from"
- "merger with"
- "acquired from"

EXAMPLES OF COMPANIES WITH FORMER NAMES:
- Meta Platforms Inc (formerly Facebook Inc, originally TheFacebook)
- Eternal Ltd (formerly called Zomata Limited)
- Wipro Ltd (formerly Western India Products Limited)
- ITC Ltd (formerly Imperial Tobacco Company)
- HDFC Bank (from Housing Development Finance Corporation)
- Google (now Alphabet Inc)
- Twitter (now X Corp)

RESPONSE FORMAT - Return ONLY this JSON structure:
{{
  "primary_alias": "Current Official Company Name",
  "aliases": [
    "Current Official Name",
    "Former Name 1", 
    "Former Name 2",
    "Common Abbreviation",
    "Trading Name",
    "Short Form"
  ],
  "stock_symbols": ["SYMBOL1", "SYMBOL2"],
  "local_variants": [
    "Company Name India",
    "Company Name Limited",
    "Company Name Pvt Ltd"
  ],
  "parent_company": "Parent Company Name if it exists or company name if it doesn't",
  "all_aliases": "Comma-separated list of all aliases",
  "confidence_score": 0.8
}}

CRITICAL REQUIREMENTS:
✓ Include ALL historical and former company names you can find
✓ Research mergers, acquisitions, rebranding history
✓ Ensure all fields are filled (use company name as default if needed)
✓ Return ONLY valid JSON without markdown or explanations

Research "{company_name}" now and provide the comprehensive JSON response:"""

        try:
            logger.info(f"🔍 Querying LLM directly for {company_name}")
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw_response = response.content
            
            logger.info(f"📄 Raw LLM response: {raw_response[:500]}...")
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            
            if json_match:
                try:
                    alias_data = json.loads(json_match.group())
                    logger.info(f"✅ Successfully parsed JSON with {len(alias_data.get('aliases', []))} aliases")
                    
                    # Validate and ensure all required fields exist
                    validated_data = self._validate_and_clean_data(alias_data, company_name, country)
                    return validated_data
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON parsing failed: {e}")
                    return self._create_fallback_data(company_name, country)
            else:
                logger.error("❌ No JSON found in LLM response")
                return self._create_fallback_data(company_name, country)
                
        except Exception as e:
            logger.error(f"❌ LLM query failed: {e}")
            return self._create_fallback_data(company_name, country)
    
    def _validate_and_clean_data(self, data: Dict[str, Any], company_name: str, country: str) -> Dict[str, Any]:
        """Validate and clean the LLM response data"""
        
        def safe_get(key: str, fallback):
            value = data.get(key)
            if not value or (isinstance(value, list) and len(value) == 0):
                return fallback
            return value
        
        # Generate programmatic variations as backup
        variations = self._generate_name_variations(company_name)
        
        # Clean and validate data
        primary_alias = safe_get("primary_alias", company_name)
        aliases = safe_get("aliases", variations)
        
        # Ensure primary alias is in aliases list
        if isinstance(aliases, list) and primary_alias not in aliases:
            aliases.insert(0, primary_alias)
        
        stock_symbols = safe_get("stock_symbols", [])
        local_variants = safe_get("local_variants", [f"{company_name} {country}"])
        parent_company = safe_get("parent_company", company_name)
        
        # Always generate adverse queries using our logic (more reliable than LLM)
        adverse_queries = self._generate_adverse_queries(primary_alias, aliases)
        
        all_aliases = safe_get("all_aliases", ", ".join(aliases[:5]))
        confidence_score = data.get("confidence_score", 0.7)
        
        return {
            "primary_alias": str(primary_alias),
            "aliases": aliases if isinstance(aliases, list) else [str(primary_alias)],
            "stock_symbols": stock_symbols if isinstance(stock_symbols, list) else [],
            "local_variants": local_variants if isinstance(local_variants, list) else [f"{company_name} {country}"],
            "parent_company": str(parent_company),
            "adverse_search_queries": adverse_queries,
            "all_aliases": str(all_aliases),
            "confidence_score": float(confidence_score) if confidence_score else 0.7
        }
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """Generate programmatic name variations"""
        variations = [name]
        
        # Common replacements
        if "Limited" in name:
            variations.append(name.replace("Limited", "Ltd"))
        if "Ltd" in name and "Limited" not in name:
            variations.append(name.replace("Ltd", "Limited"))
        if "Private" in name:
            variations.append(name.replace("Private", "Pvt"))
   
        
        # Short form without legal suffixes
        short_name = re.sub(r'\s+(Limited|Ltd|Private|Pvt|Inc|Corp|Corporation)$', '', name)
        if short_name != name:
            variations.append(short_name)
        
        return list(set(variations))
    
    def _generate_adverse_queries(self, company_name: str, aliases: List[str]) -> List[str]:
        """Generate adverse search queries"""
        keywords = ["fraud", "scandal", "lawsuit", "investigation", "penalty", 
                   "misconduct", "controversy", "fine", "violation", "accused",
                   "corruption", "bribery", "embezzlement", "criminal charges", "regulatory action"]
        
        queries = []
        
        # Primary company queries with main keywords
        for keyword in keywords[:8]:
            queries.append(f"{company_name} {keyword}")
        
        # Add queries for former names/aliases
        for alias in aliases[1:5]:  # Use up to 4 additional aliases
            if alias != company_name and len(queries) < 13:
                queries.append(f"{alias} scandal")
        
        # Fill remaining slots with additional variations
        additional_queries = [
            f"{company_name} adverse news",
            f"{company_name} legal issues",
            f"{company_name} compliance violation",
            f"{company_name} ethics violation",
            f"{company_name} regulatory fine"
        ]
        
        for query in additional_queries:
            if len(queries) < 15:
                queries.append(query)
        
        return queries[:15]
    
    def _create_fallback_data(self, company_name: str, country: str) -> Dict[str, Any]:
        """Create fallback data when LLM fails"""
        variations = self._generate_name_variations(company_name)
        
        return {
            "primary_alias": company_name,
            "aliases": variations,
            "stock_symbols": [],
            "local_variants": [f"{company_name} {country}"],
            "parent_company": company_name,
            "adverse_search_queries": self._generate_adverse_queries(company_name, variations),
            "all_aliases": ", ".join(variations),
            "confidence_score": 0.3
        }

# Global instance
alias_agent_improved = AliasGenerationImproved()
