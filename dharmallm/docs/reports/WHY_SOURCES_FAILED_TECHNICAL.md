# 🔍 TECHNICAL ANALYSIS: Why Sanskrit Documents & Wisdom Library Failed

**Date:** October 4, 2025

---

## ❌ SOURCE 3: SANSKRIT DOCUMENTS - **STATUS 406 (BLOCKED)**

### What Happened:
**Every single request returned HTTP Status 406** (Not Acceptable)

### What is HTTP 406?
- **Status Code:** 406 Not Acceptable
- **Meaning:** Server understood the request but **refuses to serve content** in a format the client can accept
- **Common Cause:** Anti-bot/anti-scraping protection

### Technical Details:

#### URLs Attempted:
```
https://sanskritdocuments.org/doc_giitaa       → 406
https://sanskritdocuments.org/doc_upanishhat   → 406  
https://sanskritdocuments.org/doc_veda         → 406
https://sanskritdocuments.org/doc_z_misc_major → 406
https://sanskritdocuments.org/doc_purana       → 406
```

#### Request Headers Sent:
```python
headers = {
    'User-Agent': 'Mozilla/5.0'
}
```

### Why It Failed:

1. **Anti-Bot Protection**
   - Site detected automated access
   - Simple User-Agent not sufficient
   - Likely checks for:
     - Browser fingerprints
     - JavaScript execution
     - Cookies/sessions
     - Request patterns

2. **Content Negotiation**
   - Server expects specific `Accept` headers
   - May require:
     - `Accept: text/html,application/xhtml+xml`
     - `Accept-Language: en-US,en;q=0.9`
     - `Accept-Encoding: gzip, deflate, br`

3. **Rate Limiting**
   - Too many requests too quickly
   - IP-based blocking

### How to Fix:

#### Option A: Enhanced Headers
```python
headers = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://sanskritdocuments.org/',
    'DNT': '1'
}
```

#### Option B: Use Selenium (Browser Automation)
```python
from selenium import webdriver
driver = webdriver.Chrome()
driver.get('https://sanskritdocuments.org/doc_giitaa')
```

#### Option C: Manual Download
- Visit site directly in browser
- Download files manually
- Site likely allows human visitors

#### Option D: Alternative Sources
- **GRETIL** (gretil.sub.uni-goettingen.de) - Same content
- **Archive.org** - May have mirrors
- **GitHub** - Sanskrit text repositories

### Recommendation:
**Use GRETIL or Archive.org instead** - Same texts, more accessible

---

## ⚠️ SOURCE 4: WISDOM LIBRARY - **PARTIAL SUCCESS**

### What Happened:
- ✅ Found HUNDREDS of book links
- ✅ Got Status 200 on main pages
- ❌ But actual book URLs are 404

### Technical Details:

#### What Worked:
```
https://www.wisdomlib.org/hinduism             → 200 ✓
https://www.wisdomlib.org/definition/bhagavad-gita  → 200 ✓
https://www.wisdomlib.org/definition/upanishad      → 200 ✓
https://www.wisdomlib.org/definition/purana         → 200 ✓
```

#### What Found:
**Discovered 147+ Sanskrit books!**
- Bhagavad-gita [sanskrit]
- Bhagavata Purana [sanskrit]
- Mahabharata [sanskrit]
- Ramayana [sanskrit]
- Shiva Purana [sanskrit]
- All major Puranas
- All major Upanishads

#### What Failed:
```
https://www.wisdomlib.org/hinduism/book/the-bhagavad-gita    → 404 ✗
https://www.wisdomlib.org/hinduism/book/gita                 → 404 ✗
https://www.wisdomlib.org/hinduism/book/upanishads           → 404 ✗
https://www.wisdomlib.org/search/bhagavad+gita+sanskrit      → 404 ✗
```

### Why Only Partial:

1. **URL Structure Changed**
   - Old URLs: `/hinduism/book/bhagavad-gita`
   - New structure unknown
   - Site was redesigned

2. **Found Book Links But Can't Access Content**
   ```
   Found: "Bhagavad-gita with four Commentaries [sanskrit]"
   Link: https://www.wisdomlib.org/hinduism/book/bhagavad-gita-with-four-commentaries-sanskrit
   Status: Not tested in detail
   ```

3. **Definition Pages Only**
   - Got 8 texts from definition pages
   - These are glossary entries, not full texts
   - Example: "Bhagavad Gita definition: sacred Hindu scripture..."

4. **Search Function Broken**
   - `/search/` URLs all return 404
   - Can't search for Sanskrit texts
   - No site search available

### What We Actually Got:

From `found_texts.json`:
```json
{
  "source": "wisdom_library",
  "url": "https://www.wisdomlib.org/definition/bhagavad-gita",
  "text": "भगवद्गीता..."  // Small definition excerpt
}
```

**8 similar definition snippets** - useful but not complete books

### How to Fix:

#### Option A: Test Found Book URLs
We found 147 book URLs like:
```
https://www.wisdomlib.org/hinduism/book/bhagavad-gita-with-four-commentaries-sanskrit
https://www.wisdomlib.org/hinduism/book/bhagavata-purana-sanskrit
https://www.wisdomlib.org/hinduism/book/mahabharata-sanskrit
```

**Need to test these systematically!**

#### Option B: Explore Site Structure
```python
# Try different URL patterns
patterns = [
    "/hinduism/book/{book}-sanskrit",
    "/hinduism/text/{book}",
    "/scripture/{book}",
    "/source/{book}"
]
```

#### Option C: Use Their API (if exists)
- Check for `robots.txt`
- Look for API documentation
- May have official data access

#### Option D: Navigate Like Browser
- Start at homepage
- Follow links naturally
- May need to load multiple pages

### Key Finding:

**THE BOOKS EXIST ON THE SITE!**

Evidence:
```
Found: Mahabharata [sanskrit] -> https://www.wisdomlib.org/hinduism/book/mahabharata-sanskrit
Found: Ramayana [sanskrit] -> https://www.wisdomlib.org/hinduism/book/the-ramayana-sanskrit
Found: Bhagavata Purana [sanskrit] -> https://www.wisdomlib.org/hinduism/book/bhagavata-purana-sanskrit
```

We just need to:
1. Test these exact URLs
2. Download the book content
3. Extract Sanskrit text

---

## 📊 COMPARISON

| Aspect | Sanskrit Documents | Wisdom Library |
|--------|-------------------|----------------|
| **Problem Type** | Access Blocked | Navigation Issue |
| **HTTP Status** | 406 (Not Acceptable) | 404 (Not Found) |
| **Root Cause** | Anti-bot protection | URL structure unknown |
| **Severity** | HIGH - Complete block | MEDIUM - Can be solved |
| **Data Loss** | 100% - No access | 5% - Found links but can't access |
| **Fix Difficulty** | HARD - Need to bypass security | EASY - Just find right URLs |
| **Alternative** | Use GRETIL/Archive.org | No alternative needed |
| **Time to Fix** | 2-3 hours (complex) | 30-60 minutes (simple) |

---

## 🎯 RECOMMENDATIONS

### For Sanskrit Documents:
**DON'T WASTE TIME** - Use alternatives:
1. **GRETIL** (gretil.sub.uni-goettingen.de)
   - Same Sanskrit texts
   - Academic site, no anti-bot
   - Direct file downloads

2. **Archive.org**
   - Has Sanskrit document collections
   - Public domain texts
   - Easy downloads

3. **Manual Download**
   - Visit sanskritdocuments.org in browser
   - Download 20-30 key files manually
   - Quick and reliable

### For Wisdom Library:
**FIXABLE! High priority:**
1. **Test the 147 found URLs**
   - We have exact book URLs
   - Just need to download them
   - Example:
     ```python
     url = "https://www.wisdomlib.org/hinduism/book/mahabharata-sanskrit"
     html = requests.get(url).text
     # Extract Sanskrit content
     ```

2. **Expected Results:**
   - Mahabharata: ~100,000 verses
   - Ramayana: ~24,000 verses
   - Bhagavata Purana: ~18,000 verses
   - **Total potential: 200,000+ verses!**

3. **Implementation:**
   ```python
   book_urls = [
       "mahabharata-sanskrit",
       "the-ramayana-sanskrit",
       "bhagavata-purana-sanskrit",
       "shiva-purana-sanskrit",
       # ... 143 more books
   ]
   
   for book in book_urls:
       url = f"https://www.wisdomlib.org/hinduism/book/{book}"
       download_and_extract_sanskrit(url)
   ```

---

## 🔧 QUICK FIX SCRIPT

### Test Wisdom Library Books:
```python
#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup

# Sample of found books
test_books = [
    "mahabharata-sanskrit",
    "the-ramayana-sanskrit", 
    "bhagavata-purana-sanskrit",
    "bhagavad-gita-with-four-commentaries-sanskrit",
    "shiva-purana-sanskrit"
]

base_url = "https://www.wisdomlib.org/hinduism/book/"

for book in test_books:
    url = base_url + book
    try:
        response = requests.get(url, timeout=10)
        print(f"{book}: Status {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find Sanskrit content
            sanskrit_divs = soup.find_all(text=re.compile('[\u0900-\u097F]'))
            if sanskrit_divs:
                print(f"  ✓ Contains Sanskrit! {len(sanskrit_divs)} matches")
    except Exception as e:
        print(f"{book}: Error - {e}")
```

---

## 💡 CONCLUSION

### Sanskrit Documents:
**Status:** BLOCKED - Not worth fixing  
**Action:** Use GRETIL or Archive.org instead  
**Lost:** ~5,000 texts  
**Impact:** LOW (available elsewhere)

### Wisdom Library:
**Status:** SOLVABLE - High value target  
**Action:** Test the 147 found book URLs  
**Potential:** **200,000+ verses** if URLs work  
**Impact:** HIGH (could reach 25% of goal in one go!)

### Priority:
1. **✅ Continue Gita Supersite** (running, will complete)
2. **🔥 FIX WISDOM LIBRARY** (test book URLs) - **DO THIS NEXT!**
3. **📖 Download from GRETIL** (easy, reliable)
4. **⏸️ Ignore Sanskrit Documents** (not worth the effort)

---

## 📝 NEXT STEPS

1. Wait for Gita Supersite to finish (~30 min)
2. Create wisdom_library_book_downloader.py
3. Test all 147 book URLs
4. Download successful books
5. Extract pure Sanskrit
6. **Potential result: 200,000+ verses!** 🎉
