# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Restaurant Lead Enrichment Tool - A Python CLI that processes restaurant business CSV/Excel files, resolves actual restaurant names (DBA) from LLC names, identifies owners, and matches against existing contact data.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the tool
python main.py input.csv -o output.csv --batch-size 10

# Run with API keys inline (if not exported)
OPENROUTER_API_KEY="sk-or-..." GOOGLE_PLACES_API_KEY="..." python main.py input.csv -o output.csv

# Clear cache to force fresh API calls
rm -rf .cache
```

## Required Environment Variables

- `GOOGLE_PLACES_API_KEY` - Google Places Nearby Search API
- `OPENROUTER_API_KEY` - OpenRouter API key (accesses Perplexity sonar-pro model)

## Architecture

Single-file CLI (`main.py`) with async parallel processing:

1. **Data Classes**: `Config`, `PersonInfo`, `RestaurantRecord` - core data structures
2. **CacheManager**: File-based MD5-hashed cache in `.cache/` to avoid duplicate API calls
3. **API Clients**:
   - `GooglePlacesClient` - Nearby Search for restaurant name resolution (uses lat/long)
   - `PerplexityClient` - Via OpenRouter for name resolution fallback and owner discovery
4. **Processing Pipeline** (`process_record`):
   - Extract DBA from LLC name if present (regex)
   - Try Google Places if coordinates exist
   - Fall back to Perplexity for name resolution
   - Find owner via Perplexity
   - Fuzzy match owner against CSV persons (rapidfuzz, 80% threshold)
5. **Async batch processing** with semaphore-limited concurrency

## Input CSV Expected Columns

`fein, name, lat, long, address, city, state, zip, phone, county, expdate, website, email1, name1-10, phone1-10`

## Output CSV Columns

`FEIN, Name, OwnerName, Address, City, State, Zip, Phone, Email, County, Expdate, Website, LLC_Name, ContactSource`

One row per input record with single best owner match.

## Test Input File

**Primary test file**: `/Users/samruben/Downloads/InsuranceX_50+.csv`

Use this file for all testing going forward:
```bash
python main.py /Users/samruben/Downloads/InsuranceX_50+.csv -o output.csv --limit 10 -v
```

## TODO: Whitepages Integration

Whitepages enrichment is currently disabled (free trial limited to 50 lookups).

### Why Whitepages is Needed

The input CSV contains **restaurant** address/phone, NOT the owner's personal info. The goal is to find where the **owner lives** for direct outreach. Whitepages provides owner's personal address, phone, and email.

### Correct Pipeline

```
Step 1: Resolve Restaurant Name
  - Google Places (if lat/long) → Perplexity fallback

Step 2: Find Owner Name
  - Perplexity: "Who owns [restaurant] in [city], [state]?"
  - Bonus: Check if owner matches name1-10 columns (may already have their phone)

Step 3: Whitepages Lookup (EVERY record)
  - Call with: owner name + city + state
  - Returns: owner's PERSONAL address, phone, email (where they LIVE)
  - This is the key enrichment step

Step 4: Output combined data
```

### To Re-enable

1. Get Whitepages Pro API key from https://pro.whitepages.com/
2. Add `WHITEPAGES_API_KEY` environment variable
3. Restore `WhitepagesClient` class:
   - Endpoint: `https://proapi.whitepages.com/3.3/person`
   - Params: `api_key`, `name`, `address.city`, `address.state_code`
4. Call Whitepages for **every** owner found (not just missing matches)
5. Output should include owner's personal phone/address, distinct from restaurant contact
