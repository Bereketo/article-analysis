import requests
import json

def test_aliases_endpoint():
    """Test the /aliases endpoint"""
    
    url = "http://localhost:8000/api/aliases"
    
    test_data = {
        "company_name": "Adani Enterprises Limited",
        "country": "India"
    }
    
    try:
        response = requests.post(url, json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            
            print("✅ API Response:")
            print(json.dumps(result, indent=2))
            
            # Validate response structure
            required_fields = [
                "primary_alias", "aliases", "stock_symbols", 
                "local_variants", "parent_company", 
                "adverse_search_queries", "all_aliases"
            ]
            
            for field in required_fields:
                if field not in result:
                    print(f"❌ Missing field: {field}")
                else:
                    print(f"✅ Field present: {field}")
            
            print(f"\n📊 Generated {len(result['adverse_search_queries'])} adverse search queries")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_aliases_endpoint()