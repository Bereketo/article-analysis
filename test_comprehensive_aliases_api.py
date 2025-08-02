import requests
import json

def test_comprehensive_aliases_endpoint():
    """Test the comprehensive /aliases endpoint"""
    
    url = "http://localhost:8000/api/aliases"
    
    test_cases = [
        {
            "company_name": "Adani Enterprises Limited",
            "country": "India"
        },
        {
            "company_name": "Infosys Limited", 
            "country": "India"
        },
        {
            "company_name": "Tata Consultancy Services",
            "country": "India"
        }
    ]
    
    for test_data in test_cases:
        print(f"\n🧪 Testing: {test_data['company_name']}")
        print("=" * 50)
        
        try:
            response = requests.post(url, json=test_data)
            
            if response.status_code == 200:
                result = response.json()
                
                print("✅ API Response:")
                print(json.dumps(result, indent=2))
                
                # Validate comprehensive response
                print(f"\n📊 Analysis:")
                print(f"✅ Primary Alias: {result['primary_alias']}")
                print(f"✅ Total Aliases: {len(result['aliases'])}")
                print(f"✅ Stock Symbols: {len(result['stock_symbols'])}")
                print(f"✅ Local Variants: {len(result['local_variants'])}")
                print(f"✅ Parent Company: {result['parent_company']}")
                print(f"✅ Adverse Queries: {len(result['adverse_search_queries'])}")
                print(f"✅ Confidence Score: {result.get('confidence_score', 'N/A')}")
                
                # Validate adverse query comprehensiveness
                adverse_queries = result['adverse_search_queries']
                priority_terms = ['scandal', 'fraud', 'investigation', 'lawsuit']
                found_priority = sum(1 for query in adverse_queries 
                                   for term in priority_terms if term in query.lower())
                
                print(f"✅ Priority adverse terms coverage: {found_priority} queries")
                
                if len(adverse_queries) >= 50:
                    print("✅ Comprehensive adverse coverage achieved")
                else:
                    print("⚠️ Limited adverse coverage")
                
            else:
                print(f"❌ API Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_comprehensive_aliases_endpoint()