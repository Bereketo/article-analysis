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
    "Former Name 2 if it exists",
    "Common Abbreviation",
    "Trading Name",
  ],
  "stock_symbols": ["SYMBOL1", "SYMBOL2"],
  "local_variants": [
    "regional name of company in english"
  ],
  "parent_company": "Parent Company Name if it exists or company name if it doesn't",
  "target_names": [
    "Current Official Name",
    "Former Name 1",
    "Common Abbreviation", 
    "SYMBOL1",
    "regional name",
    "Parent Company Name"
  ],
  "all_aliases": "Comma-separated list of all aliases",
  "confidence_score": 0.8
}}

CRITICAL REQUIREMENTS:
✓ Include ALL historical and former company names you can find
✓ Research mergers, acquisitions, rebranding history
✓ Ensure all fields are filled (use company name as default if needed)
✓ Return ONLY valid JSON without markdown or explanations
✓ Don't Include Number in Stock symbols only letters
✓ For target_names: combine ALL aliases + stock symbols + local variants + parent company (if different) into one comprehensive list
✓ target_names should be the master list of ALL possible names this company could be searched under
✓ REMOVE DUPLICATES from target_names - ensure each name appears only once (case-insensitive)


Research "{company_name}" now and provide the comprehensive JSON response:"""

        try:
            logger.info(f"🔍 Querying LLM directly for {company_name}")
            
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            raw_response = response.content
            
            logger.info(f"📄 Raw LLM response: {raw_response[:500]}...")
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
            
            if json_match:
                try:
                    alias_data = json.loads(json_match.group())
                    logger.info(f"✅ Successfully parsed JSON with {len(alias_data.get('aliases', []))} aliases")
                    logger.info(f"🎯 LLM response contains target_names: {'target_names' in alias_data}")
                    if 'target_names' in alias_data:
                        logger.info(f"📊 LLM target_names count: {len(alias_data.get('target_names', []))}")
                    
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
        
        variations = self._generate_name_variations(company_name)
        
        primary_alias = safe_get("primary_alias", company_name)
        aliases = safe_get("aliases", variations)
        
        if isinstance(aliases, list) and primary_alias not in aliases:
            aliases.insert(0, primary_alias)
        
        stock_symbols = safe_get("stock_symbols", [])
        local_variants = safe_get("local_variants", [f"{company_name} {country}"])
        parent_company = safe_get("parent_company", company_name)
        
        # Get target_names from LLM response or build them internally as fallback
        llm_target_names = data.get("target_names", [])
        if llm_target_names and isinstance(llm_target_names, list) and len(llm_target_names) > 0:
            # Use LLM-generated target_names but deduplicate them
            raw_target_names = [str(name).strip() for name in llm_target_names if name and str(name).strip()]
            target_names = self._deduplicate_target_names(raw_target_names)
            logger.info(f"🎯 Using LLM-generated target_names: {len(raw_target_names)} raw → {len(target_names)} deduplicated")
        else:
            # Fallback to building target names internally
            target_names = self._build_target_names(primary_alias, aliases, parent_company, stock_symbols, local_variants)
            logger.info(f"🔧 Using internally generated target_names: {len(target_names)} entries")
        
        adverse_queries = self._generate_adverse_queries(primary_alias, aliases, parent_company, stock_symbols, local_variants)
        
        all_aliases = safe_get("all_aliases", ", ".join(aliases[:5]))
        confidence_score = data.get("confidence_score", 0.7)
        
        return {
            "primary_alias": str(primary_alias),
            "aliases": aliases if isinstance(aliases, list) else [str(primary_alias)],
            "stock_symbols": stock_symbols if isinstance(stock_symbols, list) else [],
            "local_variants": local_variants if isinstance(local_variants, list) else [f"{company_name} {country}"],
            "target_names": target_names, 
            "parent_company": str(parent_company),
            "adverse_search_queries": adverse_queries,
            "all_aliases": str(all_aliases),
            "confidence_score": float(confidence_score) if confidence_score else 0.7
        }
    
    def _deduplicate_target_names(self, target_names: List[str]) -> List[str]:
        """Remove duplicates from target names using case-insensitive comparison"""
        
        seen_lower = set()
        deduplicated = []
        
        for name in target_names:
            if name and name.strip():
                name_clean = name.strip()
                name_lower = name_clean.lower()
                
                if name_lower not in seen_lower:
                    deduplicated.append(name_clean)
                    seen_lower.add(name_lower)
                    logger.debug(f"✓ Kept: {name_clean}")
                else:
                    logger.debug(f"⚠️  Removed duplicate: {name_clean}")
        
        return deduplicated
    
    def _generate_name_variations(self, name: str) -> List[str]:
        """Generate programmatic name variations"""
        variations = [name]
        
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
    
    def _generate_adverse_queries(self, company_name: str, aliases: List[str], parent_company: str = None, stock_symbols: List[str] = None, local_variants: List[str] = None) -> List[str]:
        """Generate comprehensive adverse search queries using structured keyword categories.
        
        Implements three comprehensive keyword categories as per specification:
        a. Fraud/Financial: fraud, default, kickback, scandal, probe, penalty, misconduct, etc.
        b. Legal/Criminal: ban, banned, politics, political, murder, bribe, rape, theft, etc.
        c. Investigation/Compliance: investigation, penalized, probed, case, criminal, manipulation, etc.
        
        Uses TARGET_NAME = alias + local_variant + parent_company + stock_symbol combinations
        with individual target names combined with keyword groups using AND operators.
        
        Example format: "Target Name" AND (keyword1 OR keyword2 OR keyword3)
        """
        
        # Ensure defaults for optional parameters
        if stock_symbols is None:
            stock_symbols = []
        if local_variants is None:
            local_variants = []
        
        # Category A: Fraud/Financial/Regulatory Keywords
        fraud_financial_keywords = [
            "fraud", "default", "kickback", "scandal", "probe", "penalty", "misconduct",
            "CBI", "imprisonment", "police", "scam", "vigilance", "litigation", "arrests",
            "accused", "accuse", "alleged", "allegedly", "sentenced", "illegal", "lawsuit"
        ]
        
        # Category B: Legal/Criminal/Political Keywords
        legal_criminal_keywords = [
            "ban", "banned", "politics", "political", "murder", "bribe", "bribed",
            "rape", "raped", "raping", "robbed", "rob", "theft", "charged", "mobbed",
            "jail", "jailed", "victim", "underworld", "terrorists", "terrorist", "terrorism"
        ]
        
        # Category C: Investigation/Compliance/Manipulation Keywords
        investigation_compliance_keywords = [
            "investigate", "investigated", "investigation", "penalized", "probed", "case",
            "criminal", "crime", "abuse", "abused", "strike", "accident", "manipulated",
            "manipulation", "downfall", "loss", "cronies", "absconds", "absconding",
            "absconded", "lobbying", "compliance", "complying", "prison", "imprisoned",
            "warned", "warn", "warning", "terminate"
        ]
        
        # Build comprehensive TARGET_NAME combinations as per specification
        target_names = self._build_target_names(company_name, aliases, parent_company, stock_symbols, local_variants)
        print(target_names)
        logger.info(f"🎯 Built {len(target_names)} target name combinations: {target_names[:5]}...")
        
        # Generate OR/AND structured queries
        queries = self._generate_structured_queries(
            target_names, 
            fraud_financial_keywords, 
            legal_criminal_keywords, 
            investigation_compliance_keywords
        )
        
        return queries  # Comprehensive coverage - no artificial limit
    
    def _build_target_names(self, company_name: str, aliases: List[str], parent_company: str, stock_symbols: List[str], local_variants: List[str]) -> List[str]:
        """Build TARGET_NAME combinations: alias + local_variant + parent_company + stock_symbol"""
        
        target_names = []
        added_names_lower = set()  # Track lowercase versions to avoid exact duplicates
        
        # Helper function to add target if not duplicate
        def add_target(name: str):
            if name and name.strip():
                name_clean = name.strip()
                name_lower = name_clean.lower()
                # Only check for exact duplicates (case-insensitive), but allow similar names
                if name_lower not in added_names_lower:
                    target_names.append(name_clean)
                    added_names_lower.add(name_lower)
                    logger.debug(f"✓ Added target: {name_clean}")
                else:
                    logger.debug(f"⚠️  Skipped duplicate: {name_clean}")
        
        # 1. Primary company name first
        add_target(company_name)
        
        # 2. All aliases (don't be restrictive here!)
        logger.info(f"📝 Processing {len(aliases)} aliases: {aliases}")
        for alias in aliases:
            add_target(alias)
        
        # 3. Parent company (if different and valid)
        if (parent_company and 
            parent_company.lower() not in ["unknown", "n/a", "none", "", company_name.lower()]):
            add_target(parent_company)
            logger.info(f"🏢 Added parent company: {parent_company}")
        
        # 4. Stock symbols
        logger.info(f"💹 Processing {len(stock_symbols)} stock symbols: {stock_symbols}")
        for symbol in stock_symbols:
            add_target(symbol)
        
        # 5. Local variants
        logger.info(f"🌍 Processing {len(local_variants)} local variants: {local_variants}")
        for variant in local_variants:
            # Don't add generic "Company Name Country" patterns unless they're meaningful
            if variant and variant != f"{company_name} India":
                add_target(variant)
        
        logger.info(f"🎯 Built {len(target_names)} unique target names: {target_names}")
        logger.info(f"📊 Distribution: Primary=1, Aliases={len([a for a in aliases if a])}, Parent={'1' if parent_company and parent_company.lower() not in ['unknown', 'n/a', 'none', company_name.lower()] else '0'}, Symbols={len(stock_symbols)}, Variants={len([v for v in local_variants if v and v != f'{company_name} India'])}")
        
        return target_names[:12]  # Increased limit to allow more comprehensive coverage
    
    def _generate_structured_queries(self, target_names: List[str], fraud_keywords: List[str], legal_keywords: List[str], investigation_keywords: List[str]) -> List[str]:
        """Generate structured OR/AND queries as per specification examples"""
        
        queries = []
        
        # Optimized keyword groups - 3 comprehensive groups with 15 keywords each
        keyword_groups = [
            # Group 1: Fraud, Financial Crimes & Corruption (15 keywords)
            ["fraud", "scandal", "kickback", "misconduct", "scam", "bribe", "corruption", 
             "embezzlement", "money-laundering", "forgery", "default", "bankruptcy", 
             "insolvency", "penalty", "fine"],
            
            # Group 2: Legal, Criminal & Violent Crimes (15 keywords)  
            ["lawsuit", "litigation", "investigation", "probe", "arrested", "charged", 
             "accused", "criminal", "police", "CBI", "murder", "rape", "assault", 
             "violence", "terrorism"],
            
            # Group 3: Regulatory, Compliance & Consequences (15 keywords)
            ["banned", "suspension", "sanctions", "violation", "breach", "imprisonment", 
             "prison", "jail", "jailed", "sentenced", "conviction", "guilty", "manipulated", 
             "manipulation", "compliance"]
        ]
        
        # COMPREHENSIVE APPROACH: Every target name with every keyword group
        logger.info(f"🚀 Generating comprehensive queries: {len(target_names)} targets × {len(keyword_groups)} keyword groups")
        
        # Generate every combination of target names with keyword groups (single target only)
        for target_name in target_names:
            for keyword_group in keyword_groups:
                # Build target part - single target only
                target_part = f'"{target_name}"'
                
                # Build keyword part with OR
                keyword_part = f"({' OR '.join(keyword_group)})"
                
                # Combine with AND
                query = f"{target_part} AND {keyword_part}"
                queries.append(query)
        
        logger.info(f"🔍 Generated {len(queries)} structured OR/AND queries")
        return queries
    
    def _create_fallback_data(self, company_name: str, country: str) -> Dict[str, Any]:
        """Create fallback data when LLM fails"""
        variations = self._generate_name_variations(company_name)
        target_names = self._build_target_names(company_name, variations, company_name, [], [f"{company_name} {country}"])
        
        return {
            "primary_alias": company_name,
            "aliases": variations,
            "stock_symbols": [],
            "local_variants": [f"{company_name} {country}"],
            "target_names": target_names,
            "parent_company": company_name,
            "adverse_search_queries": self._generate_adverse_queries(company_name, variations, company_name, [], [f"{company_name} {country}"]),
            "all_aliases": ", ".join(variations),
            "confidence_score": 0.3
        }

alias_agent_improved = AliasGenerationImproved()
