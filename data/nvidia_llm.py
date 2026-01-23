"""
NVIDIA LLM Integration for Company Overview
Uses Mistral Large 3 model via NVIDIA API
"""

import requests
import os

# NVIDIA API Configuration
def _get_nvidia_api_key():
    """Read NVIDIA API key from keys.txt file"""
    try:
        keys_path = os.path.join(os.path.dirname(__file__), '..', 'keys.txt')
        with open(keys_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'nvidia llms' in line.lower():
                    # Get the next non-empty line
                    if i + 1 < len(lines):
                        return lines[i + 1].strip()
        return None
    except Exception as e:
        print(f"Error reading NVIDIA API key: {e}")
        return None

NVIDIA_API_KEY = _get_nvidia_api_key()
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def get_company_overview_llm(company_name: str, symbol: str) -> str:
    """
    Get a comprehensive company overview using NVIDIA's Mistral Large 3 LLM
    
    Args:
        company_name: Full company name (e.g., "Apple Inc.")
        symbol: Stock ticker symbol (e.g., "AAPL")
    
    Returns:
        String containing LLM-generated company overview
    """
    
    if not NVIDIA_API_KEY:
        return "❌ NVIDIA API key not found in keys.txt. Please add it under '# nvidia llms' section."
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json"
    }
    
    prompt = f"""Can you give me a concise overview of {company_name} ({symbol})? 

Please include:
1. What the company does (main products/services)
2. Key business segments
3. Market position and competitive advantages
4. Recent notable developments or strategic focus

Keep it concise (3-4 paragraphs) and investor-focused."""

    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024,
        "temperature": 0.2,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        print(f"🤖 Generating AI overview for {company_name}...")
        print(f"   Using model: meta/llama-3.1-70b-instruct")
        print(f"   API endpoint: {INVOKE_URL}")
        
        response = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=15)  # Reduced to 15s
        
        # Check for errors
        if response.status_code == 403:
            print("❌ 403 Forbidden - API key may be invalid or expired")
            print("   Please verify the NVIDIA API key in keys.txt")
            print("   Get a new key at: https://build.nvidia.com/")
            print(f"   Response: {response.text[:200]}")
            return None
        elif response.status_code == 429:
            print("❌ 429 Rate Limit - Too many requests")
            return None
        elif response.status_code != 200:
            print(f"❌ HTTP {response.status_code} Error")
            print(f"   Response: {response.text[:200]}")
            return None
        
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            overview = result['choices'][0]['message']['content']
            print(f"✓ Generated {len(overview)} character overview")
            return overview
        else:
            print("⚠️  No content in LLM response")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ NVIDIA LLM request timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ NVIDIA LLM error: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return None


def get_company_overview_streaming(company_name: str, symbol: str):
    """
    Get company overview with streaming response (for real-time display)
    
    Args:
        company_name: Full company name
        symbol: Stock ticker symbol
    
    Yields:
        Chunks of text as they arrive from the LLM
    """
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "text/event-stream"
    }
    
    prompt = f"""Can you give me a concise overview of {company_name} ({symbol})? 

Please include:
1. What the company does (main products/services)
2. Key business segments
3. Market position and competitive advantages
4. Recent notable developments or strategic focus

Keep it concise (3-4 paragraphs) and investor-focused."""

    payload = {
        "model": "mistralai/mistral-large-3-675b-instruct-2512",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.15,
        "top_p": 1.00,
        "frequency_penalty": 0.00,
        "presence_penalty": 0.00,
        "stream": True
    }
    
    try:
        print(f"🤖 Streaming AI overview for {company_name}...")
        response = requests.post(INVOKE_URL, headers=headers, json=payload, stream=True, timeout=30)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                yield line.decode("utf-8")
                
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        yield None


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
        company = sys.argv[2] if len(sys.argv) > 2 else symbol
    else:
        symbol = "NVDA"
        company = "NVIDIA Corporation"
    
    print(f"\n{'='*70}")
    print(f"Testing NVIDIA LLM for {company} ({symbol})")
    print(f"{'='*70}\n")
    
    overview = get_company_overview_llm(company, symbol)
    
    if overview:
        print("\n" + "="*70)
        print("COMPANY OVERVIEW")
        print("="*70)
        print(overview)
        print("="*70)
    else:
        print("\n❌ Failed to generate overview")
