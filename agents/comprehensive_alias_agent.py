import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.bing_search import BingSearchResults
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.schema import HumanMessage

@dataclass
class ComprehensiveAliasResult:
    primary_alias: str
    aliases: List[str]
    stock_symbols: List[str]
    local_variants: List[str]
    parent_company: str
    adverse_search_queries: List[str]
    all_aliases: str
    confidence_score: float
    sources: List[str]

class ComprehensiveAliasAgent:
    """Comprehensive alias generation agent for adverse media screening"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_llm()
        self._setup_search_tools()
        
    def _setup_llm(self):
        """Setup Azure OpenAI"""
        self.llm = AzureChatOpenAI(
            openai_api_key="1f5d1bb6920844248ea17f61f73f82ac",
            azure_endpoint="https://ai-gpt-echo.openai.azure.com",
            azure_deployment="o3-mini",
            openai_api_version="2024-12-01-preview",
            max_completion_tokens=2000
        )
        
    def _setup_search_tools(self):
        """Setup search tools"""
        bing_wrapper = BingSearchAPIWrapper(
            bing_subscription_key="822e2402879b4c78b00434c7f0f4c201",
            search_kwargs={"mkt": "en-IN", "count": 20}
        )
        self.bing_tool = BingSearchResults(api_wrapper=bing_wrapper, num_results=20)
    
    async def generate_comprehensive_aliases(
        self, 
        company_name: str, 
        country: str = "India"
    ) -> ComprehensiveAliasResult:
        """Generate comprehensive company aliases"""
        
        self.logger.info(f"🎯 Starting alias generation for: {company_name}")
        
        # Step 1: Research
        research_data = await self._conduct_research(company_name, country)
        
        # Step 2: Extract alias information
        alias_data = await self._extract_alias_information(research_data, company_name)
        
        # Step 3: Generate adverse queries
        adverse_queries = self._generate_adverse_queries(alias_data)
        
        # Step 4: Structure result
        result = self._structure_result(alias_data, adverse_queries)
        
        self.logger.info(f"✅ Generated {len(result.aliases)} aliases and {len(result.adverse_search_queries)} adverse queries")
        
        return result
    
    async def _conduct_research(self, company_name: str, country: str) -> Dict[str, Any]:
        """Conduct research"""
        
        research_queries = [
            f'"{company_name}" company profile {country}',
            f'"{company_name}" stock symbol ticker',
            f'"{company_name}" parent company',
            f'"{company_name}" official name'
        ]
        
        all_research_data = []
        
        for query in research_queries:
            try:
                search_results = self.bing_tool.run(query)
                all_research_data.append({
                    "query": query,
                    "results": search_results
                })
                await asyncio.sleep(0.5)
                
            except Exception as e:
                self.logger.error(f"Search failed for query '{query}': {e}")
        
        return {
            "company_name": company_name,
            "country": country,
            "research_data": all_research_data
        }
    
    async def _extract_alias_information(self, research_data: Dict[str, Any], company_name: str) -> Dict[str, Any]:
        """Extract alias information using LLM"""
        
        combined_research = "\n\n".join([
            f"Query: {item['query']}\nResults: {item['results']}"
            for item in research_data["research_data"]
        ])

        extraction_prompt = f""" You are a research assistant specialized in finding accurate and up-to-date information.
When searching for information, use specific search terms and analyze the results carefully. Analyze the search results for "{company_name}" and extract company information.

Research Results:
{combined_research}

Extract information for "{company_name}" specifically and return in this JSON format:
{{
    "primary_alias": "{company_name}",
    "aliases": ["{company_name}", "variations of the name", "short forms", "abbreviations",  "old or former company names"],
    "stock_symbols": ["SYMBOL"],
    "local_variants": ["local language names", "regional names"],
    "parent_company": "Parent Company Name",
    "confidence_score": 0.8
}}

IMPORTANT INSTRUCTIONS:
- Only extract information about "{company_name}", not other companies
- For aliases: Include all possible variations including short forms, abbreviations, and any former or old company names
- For stock_symbols: Look for NSE/BSE ticker symbols (usually 3-10 characters, letters only)
- Stock symbols are like "ADANIENT", "RELIANCE", "TCS", "INFY" - NOT numbers
- If no stock symbol found, return empty array []
- Include ALL possible name variations in aliases"""
        
        try:
            response = await self.llm.ainvoke([HumanMessage(content=extraction_prompt)])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                alias_data = json.loads(json_match.group())
                alias_data = self._validate_alias_data(alias_data, company_name)
                
                # Additional stock symbol detection if empty
                if not alias_data.get("stock_symbols"):
                    detected_symbols = self._detect_stock_symbols(combined_research, company_name)
                    alias_data["stock_symbols"] = detected_symbols
            
                return alias_data
        
        except Exception as e:
            self.logger.error(f"LLM extraction failed: {e}")
        
        # Fallback
        return self._create_fallback_alias_data(company_name)

    def _validate_alias_data(self, data: Dict[str, Any], original_name: str) -> Dict[str, Any]:
        """Validate and clean extracted alias data"""
        
        # Ensure primary_alias is set correctly
        data["primary_alias"] = original_name
        
        # Ensure aliases list includes primary alias
        aliases = data.get("aliases", [])
        if original_name not in aliases:
            aliases.insert(0, original_name)
        
        # Add common variations if not present
        variations = self._generate_name_variations(original_name)
        for var in variations:
            if var not in aliases:
                aliases.append(var)
        
        # Remove duplicates
        data["aliases"] = list(dict.fromkeys(aliases))
        
        # Validate stock symbols
        stock_symbols = data.get("stock_symbols", [])
        validated_symbols = []
        for symbol in stock_symbols:
            if isinstance(symbol, str) and len(symbol) <= 10 and symbol.replace(".", "").isalnum():
                validated_symbols.append(symbol.upper())
        data["stock_symbols"] = validated_symbols
        
        # Enhance local variants - don't leave empty
        local_variants = data.get("local_variants", [])
        if not local_variants or local_variants == [original_name]:
            # Generate better local variants
            local_variants = self._generate_local_variants(original_name)
        data["local_variants"] = local_variants
        
        # Set defaults
        data.setdefault("parent_company", "Unknown")
        data.setdefault("confidence_score", 0.7)
        
        return data

    def _generate_name_variations(self, company_name: str) -> List[str]:
        """Generate common name variations"""
        variations = []
        
        # Common replacements
        if "Limited" in company_name:
            variations.append(company_name.replace("Limited", "Ltd"))
        if "Ltd" in company_name and "Limited" not in company_name:
            variations.append(company_name.replace("Ltd", "Limited"))
        if "Private" in company_name:
            variations.append(company_name.replace("Private", "Pvt"))
        if "Pvt" in company_name and "Private" not in company_name:
            variations.append(company_name.replace("Pvt", "Private"))
        
        # Short form
        short_name = re.sub(r'\s+(Limited|Ltd|Private|Pvt|Inc|Corp|Corporation)$', '', company_name)
        if short_name != company_name:
            variations.append(short_name)
        
        # Acronym if multiple words
        words = company_name.split()
        if len(words) > 1:
            acronym = ''.join([word[0].upper() for word in words if word not in ['Limited', 'Ltd', 'Private', 'Pvt']])
            if len(acronym) >= 2:
                variations.append(acronym)
        
        return variations

    def _generate_local_variants(self, company_name: str) -> List[str]:
        """Generate local language variants in English"""
        local_variants = [company_name]
        
        # Common Indian company translations in English
        if "Enterprises" in company_name:
            local_variants.append(company_name.replace("Enterprises", "Udyam"))
        if "Limited" in company_name:
            local_variants.append(company_name.replace("Limited", "Limited India"))
        if "Industries" in company_name:
            local_variants.append(company_name.replace("Industries", "Udyog"))
        if "Corporation" in company_name:
            local_variants.append(company_name.replace("Corporation", "Nigam"))
        
        # Add regional variants for major companies in English
        company_lower = company_name.lower()
        if "adani" in company_lower:
            local_variants.extend(["Adani Enterprises India", "Adani Group Enterprise"])
        elif "tata" in company_lower:
            local_variants.extend(["Tata India", "Tata Group"])
        elif "reliance" in company_lower:
            local_variants.extend(["Reliance India", "RIL India"])
        elif "infosys" in company_lower:
            local_variants.extend(["Infosys India", "Infosys Technologies"])
        
        
        return list(set(local_variants))
    
    def _create_fallback_alias_data(self, company_name: str) -> Dict[str, Any]:
        """Create fallback alias data"""
        
        basic_aliases = [company_name]
        
        # Add common variations
        if "Limited" in company_name:
            basic_aliases.append(company_name.replace("Limited", "Ltd"))
        if "Ltd" in company_name and "Limited" not in company_name:
            basic_aliases.append(company_name.replace("Ltd", "Limited"))
        if "Private" in company_name:
            basic_aliases.append(company_name.replace("Private", "Pvt"))
        
        # Short form
        short_name = re.sub(r'\s+(Limited|Ltd|Private|Pvt|Inc|Corp|Corporation)$', '', company_name)
        if short_name != company_name:
            basic_aliases.append(short_name)
        
        return {
            "primary_alias": company_name,
            "aliases": list(set(basic_aliases)),
            "stock_symbols": [],
            "local_variants": [company_name],
            "parent_company": "Unknown",
            "confidence_score": 0.5
        }
    
    def _generate_adverse_queries(self, alias_data: Dict[str, Any]) -> List[str]:
        """Generate 25 adverse search queries using comprehensive keyword combinations"""
        
        # Comprehensive adverse keywords as provided
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
        
        adverse_queries = []
        all_names = alias_data["aliases"][:3]  # Top 3 aliases
        
        # 1. High priority keywords (8 queries)
        priority_keywords = ["fraud", "scandal", "investigation", "lawsuit", "penalty", "bribe", "corruption", "CBI"]
        primary_name = alias_data["primary_alias"]
        for keyword in priority_keywords:
            adverse_queries.append(f"{primary_name} {keyword}")
        
        # 2. Essential keywords for all aliases (9 queries)
        essential_keywords = ["fraud", "scandal", "investigation"]
        for name in all_names:
            for keyword in essential_keywords:
                adverse_queries.append(f"{name} {keyword}")
        
        # 3. Additional comprehensive keywords (5 queries)
        additional_keywords = ["misconduct", "penalty", "litigation", "accused", "banned"]
        for i, keyword in enumerate(additional_keywords):
            if i < len(all_names):
                adverse_queries.append(f"{all_names[i]} {keyword}")
        
        # 4. Parent company searches (2 queries)
        if alias_data["parent_company"] != "Unknown":
            parent = alias_data["parent_company"]
            for keyword in ["scandal", "fraud"]:
                adverse_queries.append(f"{parent} {keyword}")
        
        # 5. Stock symbol searches (1 query)
        for symbol in alias_data.get("stock_symbols", [])[:1]:
            adverse_queries.append(f"{symbol} fraud")
        
        # Remove duplicates and limit to 25
        seen = set()
        unique_queries = []
        for query in adverse_queries:
            query_lower = query.lower()
            if query_lower not in seen:
                seen.add(query_lower)
                unique_queries.append(query)
        
        return unique_queries[:25]
    
    def _structure_result(self, alias_data: Dict[str, Any], adverse_queries: List[str]) -> ComprehensiveAliasResult:
        """Structure final result"""
        
        all_aliases_str = ", ".join(alias_data["aliases"])
        
        self.logger.info(f"📊 Results: {len(alias_data['aliases'])} aliases, {len(adverse_queries)} adverse queries")
        
        return ComprehensiveAliasResult(
            primary_alias=alias_data["primary_alias"],
            aliases=alias_data["aliases"],
            stock_symbols=alias_data["stock_symbols"],
            local_variants=alias_data["local_variants"],
            parent_company=alias_data["parent_company"],
            adverse_search_queries=adverse_queries,
            all_aliases=all_aliases_str,
            confidence_score=alias_data["confidence_score"],
            sources=[]
        )

    def _detect_stock_symbols(self, research_text: str, company_name: str) -> List[str]:
        """Detect stock symbols from research text using patterns"""
        
        symbols = []
        
        # Common patterns for Indian stock symbols
        patterns = [
            r'NSE:\s*([A-Z]{3,10})',
            r'BSE:\s*([A-Z]{3,10})',
            r'Symbol:\s*([A-Z]{3,10})',
            r'Ticker:\s*([A-Z]{3,10})',
            r'Stock Code:\s*([A-Z]{3,10})',
            r'\b([A-Z]{3,10})\s*(?:NSE|BSE)',
            r'(?:NSE|BSE)\s*:\s*([A-Z]{3,10})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, research_text, re.IGNORECASE)
            for match in matches:
                symbol = match.upper().strip()
                if self._is_valid_stock_symbol(symbol, company_name):
                    symbols.append(symbol)
        
        # Known mappings for major companies
        company_lower = company_name.lower()
        known_symbols = {
            "adani enterprises": "ADANIENT",
            "adani ports": "ADANIPORTS", 
            "reliance industries": "RELIANCE",
            "tata consultancy": "TCS",
            "infosys": "INFY",
            "wipro": "WIPRO",
            "hdfc bank": "HDFCBANK",
            "icici bank": "ICICIBANK",
            "bharti airtel": "BHARTIARTL",
            "itc": "ITC"
        }
        
        for company_key, symbol in known_symbols.items():
            if company_key in company_lower:
                symbols.append(symbol)
                break
        
        return list(set(symbols))

    def _is_valid_stock_symbol(self, symbol: str, company_name: str) -> bool:
        """Validate if a string is a valid stock symbol"""
        
        # Basic validation
        if not symbol or len(symbol) < 2 or len(symbol) > 12:
            return False
        
        # Must be alphabetic (no numbers for Indian stocks)
        if not symbol.isalpha():
            return False
        
        # Exclude common false positives
        excluded = {"THE", "AND", "FOR", "WITH", "FROM", "THIS", "THAT", "COMPANY", "LIMITED", "LTD"}
        if symbol in excluded:
            return False
        
        return True

# Test function
async def test_agent():
    agent = ComprehensiveAliasAgent()
    result = await agent.generate_comprehensive_aliases("Adani Enterprises Limited", "India")
    
    print(f"Primary Alias: {result.primary_alias}")
    print(f"Total Aliases: {len(result.aliases)}")
    print(f"Stock Symbols: {result.stock_symbols}")
    print(f"Adverse Queries: {len(result.adverse_search_queries)}")
    print(f"Confidence: {result.confidence_score}")

if __name__ == "__main__":
    asyncio.run(test_agent())
