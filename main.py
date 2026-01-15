#!/usr/bin/env python3
"""
Restaurant Lead Enrichment Tool

Processes restaurant business CSV files, finds actual restaurant names (DBA) from LLC names,
identifies owners via Perplexity (through OpenRouter), and matches against existing contacts.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
from openai import AsyncOpenAI
from rapidfuzz import fuzz
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm


# Global verbose flag for debugging
VERBOSE = False


def log_verbose(msg: str) -> None:
    """Print message if verbose mode is enabled."""
    if VERBOSE:
        print(f"[DEBUG] {msg}")


# ============================================================================
# Custom Exceptions for API Error Handling
# ============================================================================


class APIError(Exception):
    """Base exception for API-related errors.
    
    Attributes:
        api_name: Name of the API that raised the error (e.g., 'Google Places', 'Perplexity').
        status_code: HTTP status code if applicable.
    """
    
    def __init__(self, message: str, api_name: str = "", status_code: Optional[int] = None):
        self.api_name = api_name
        self.status_code = status_code
        super().__init__(f"[{api_name}] {message}" if api_name else message)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded (HTTP 429)."""
    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails (HTTP 401/403)."""
    pass


class NetworkError(APIError):
    """Raised when network request fails due to connection issues."""
    pass


def handle_api_error(e: Exception, api_name: str) -> None:
    """Handle and categorize API errors with appropriate logging.
    
    Converts aiohttp exceptions into custom exception types and logs
    the error using log_verbose for debugging purposes.
    
    Args:
        e: The exception that was raised.
        api_name: Name of the API for context in error messages.
        
    Returns:
        None - allows for graceful degradation by returning None from callers.
        
    Raises:
        RateLimitError: When HTTP 429 status is detected.
        AuthenticationError: When HTTP 401/403 status is detected.
        NetworkError: When connection-level errors occur.
        APIError: For other API-related errors.
    """
    error_msg = str(e)
    
    # Handle aiohttp-specific exceptions
    if isinstance(e, aiohttp.ClientResponseError):
        if e.status == 429:
            log_verbose(f"[{api_name}] Rate limit exceeded: {error_msg}")
            raise RateLimitError(error_msg, api_name, status_code=429) from e
        elif e.status in (401, 403):
            log_verbose(f"[{api_name}] Authentication failed: {error_msg}")
            raise AuthenticationError(error_msg, api_name, status_code=e.status) from e
        else:
            log_verbose(f"[{api_name}] HTTP error {e.status}: {error_msg}")
            raise APIError(error_msg, api_name, status_code=e.status) from e
    
    elif isinstance(e, aiohttp.ClientConnectorError):
        log_verbose(f"[{api_name}] Connection error: {error_msg}")
        raise NetworkError(f"Connection failed: {error_msg}", api_name) from e
    
    elif isinstance(e, aiohttp.ClientError):
        log_verbose(f"[{api_name}] Client error: {error_msg}")
        raise NetworkError(error_msg, api_name) from e
    
    elif isinstance(e, asyncio.TimeoutError):
        log_verbose(f"[{api_name}] Request timeout: {error_msg}")
        raise NetworkError(f"Request timed out: {error_msg}", api_name) from e
    
    else:
        log_verbose(f"[{api_name}] Unexpected error: {error_msg}")
        raise APIError(error_msg, api_name) from e


async def api_request_with_fallback(
    callables: list,
    api_name: str = "API"
) -> Optional[dict]:
    """Execute API requests with fallback support.
    
    Tries each callable in order until one succeeds. This enables
    graceful degradation when primary API sources are unavailable.
    
    Args:
        callables: List of async callables (coroutine functions or awaitables)
            to try in order. Each should return a result or None.
        api_name: Name of the API chain for logging context.
        
    Returns:
        The first successful non-None result, or None if all fail.
        
    Example:
        result = await api_request_with_fallback([
            lambda: google_client.search_nearby(session, lat, lng),
            lambda: perplexity_client.resolve_restaurant_name(llc_name, ...),
        ], api_name="Restaurant Lookup")
    """
    for i, callable_fn in enumerate(callables):
        try:
            # Handle both coroutine functions and awaitables
            if callable(callable_fn):
                result = await callable_fn()
            else:
                result = await callable_fn
            
            if result is not None:
                log_verbose(f"[{api_name}] Fallback {i + 1} succeeded")
                return result
                
        except (APIError, RateLimitError, AuthenticationError, NetworkError) as e:
            log_verbose(f"[{api_name}] Fallback {i + 1} failed: {e}")
            continue
        except Exception as e:
            log_verbose(f"[{api_name}] Fallback {i + 1} unexpected error: {e}")
            continue
    
    log_verbose(f"[{api_name}] All fallbacks exhausted, returning None")
    return None


# ============================================================================
# Configuration & Data Classes
# ============================================================================


@dataclass
class Config:
    """Application configuration from environment variables."""
    google_places_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_PLACES_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    whitepages_api_key: str = field(default_factory=lambda: os.getenv("WHITEPAGES_API_KEY", ""))
    yelp_api_key: str = field(default_factory=lambda: os.getenv("YELP_API_KEY", ""))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    def __post_init__(self):
        self.cache_dir.mkdir(exist_ok=True)

    def validate(self):
        """Validate required API keys are present."""
        missing = []
        optional_missing = []

        # Required keys
        if not self.google_places_api_key:
            missing.append("GOOGLE_PLACES_API_KEY")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")

        # Optional keys (enhance results but not required)
        if not self.whitepages_api_key:
            optional_missing.append("WHITEPAGES_API_KEY")
        if not self.yelp_api_key:
            optional_missing.append("YELP_API_KEY")

        if missing:
            print(f"Error: Missing required API keys: {', '.join(missing)}")
            print("These keys are required for basic operation.")

        if optional_missing:
            print(f"Note: Optional API keys not set: {', '.join(optional_missing)}")
            print("Additional data sources will be skipped.")


@dataclass
class PersonInfo:
    """Person contact information."""
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    source: str = "csv"  # csv, whitepages, or perplexity
    # Owner's personal address from Whitepages (distinct from restaurant address)
    personal_address: Optional[str] = None  # Owner's home address (street)
    personal_city: Optional[str] = None
    personal_state: Optional[str] = None
    personal_zip: Optional[str] = None
    personal_phone: Optional[str] = None  # Owner's personal phone (may differ from CSV match)
    personal_email: Optional[str] = None  # Owner's personal email


@dataclass
class OwnerResult:
    """Result from owner discovery with confidence scoring."""
    name: str
    confidence: float  # 0.0 to 1.0
    strategy: str  # "primary", "founder", "llc_lookup", "csv_match"
    matched_csv_person: Optional["PersonInfo"] = None
    
    def __str__(self) -> str:
        return f"{self.name} (confidence={self.confidence:.2f}, strategy={self.strategy})"


@dataclass
class RestaurantRecord:
    """Processed restaurant record."""
    fein: str = ""
    llc_name: str = ""
    restaurant_name: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone: str = ""
    email: str = ""
    county: str = ""
    expdate: str = ""
    website: str = ""
    owners: list[PersonInfo] = field(default_factory=list)
    persons_from_csv: list[PersonInfo] = field(default_factory=list)


# ============================================================================
# Cache Manager
# ============================================================================

class CacheManager:
    """Simple file-based cache for API responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, prefix: str, *args) -> str:
        """Generate a cache key from arguments."""
        key_data = f"{prefix}:{':'.join(str(a) for a in args)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, prefix: str, *args) -> Optional[dict]:
        """Get cached value if exists."""
        key = self._get_cache_key(prefix, *args)
        path = self._get_cache_path(key)
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def set(self, prefix: str, *args, value: dict):
        """Cache a value."""
        key = self._get_cache_key(prefix, *args)
        path = self._get_cache_path(key)
        with open(path, "w") as f:
            json.dump(value, f)


# ============================================================================
# API Clients
# ============================================================================

class GooglePlacesClient:
    """Google Places API client for restaurant name resolution.
    
    Supports two search methods:
    - Text Search (New): Best for matching LLC names with addresses
    - Nearby Search: Fallback when only coordinates are available
    """

    def __init__(self, api_key: str, cache: CacheManager):
        self.api_key = api_key
        self.cache = cache
        self.nearby_search_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        self.text_search_url = "https://places.googleapis.com/v1/places:searchText"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_by_text(
        self,
        session: aiohttp.ClientSession,
        llc_name: str,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None
    ) -> Optional[dict]:
        """Search for restaurant using Text Search (New) API.
        
        This method is preferred when we have LLC name and address info,
        as it can match business names against Google's database.
        
        Args:
            session: aiohttp client session
            llc_name: Business/LLC name to search for
            address: Street address (optional but improves accuracy)
            city: City name (optional but improves accuracy)
            state: State code (optional but improves accuracy)
            lat: Latitude for location bias (optional)
            lng: Longitude for location bias (optional)
            
        Returns:
            dict with 'name', 'place_id', 'address' or None if not found
        """
        if not self.api_key:
            return None
        
        if not llc_name:
            return None

        # Build query string from available components
        query_parts = [llc_name]
        if address:
            query_parts.append(address)
        if city:
            query_parts.append(city)
        if state:
            query_parts.append(state)
        text_query = " ".join(query_parts)

        # Check cache first
        cached = self.cache.get("google_text_search", text_query)
        if cached:
            return cached

        # Build request headers with API key and field mask
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.api_key,
            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.id"
        }

        # Build request body
        body: dict = {
            "textQuery": text_query,
            "includedType": "restaurant"
        }

        # Add location bias if coordinates available (500m radius circle)
        if lat is not None and lng is not None:
            body["locationBias"] = {
                "circle": {
                    "center": {
                        "latitude": lat,
                        "longitude": lng
                    },
                    "radius": 500.0
                }
            }

        try:
            async with session.post(self.text_search_url, headers=headers, json=body) as resp:
                if resp.status == 429:
                    # Rate limited - let retry handle it
                    raise Exception("Google Places API rate limited")
                
                if resp.status != 200:
                    # Log non-200 responses for debugging
                    error_text = await resp.text()
                    print(f"Google Text Search API error {resp.status}: {error_text[:200]}")
                    return None
                
                data = await resp.json()

                # Extract first result if available
                places = data.get("places", [])
                if places:
                    first_place = places[0]
                    result = {
                        "name": first_place.get("displayName", {}).get("text"),
                        "place_id": first_place.get("id"),
                        "address": first_place.get("formattedAddress")
                    }
                    if result["name"]:
                        self.cache.set("google_text_search", text_query, value=result)
                        return result

        except aiohttp.ClientError as e:
            print(f"Google Text Search request failed: {e}")
            raise  # Let retry decorator handle it

        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_nearby(
        self,
        session: aiohttp.ClientSession,
        lat: float,
        lng: float,
        radius: int = 50
    ) -> Optional[dict]:
        """Find restaurant at given coordinates using Nearby Search API.
        
        This is the fallback method when only coordinates are available.
        
        Args:
            session: aiohttp client session
            lat: Latitude coordinate
            lng: Longitude coordinate  
            radius: Search radius in meters (default 50)
            
        Returns:
            dict with 'name', 'place_id', 'address' or None if not found
        """
        if not self.api_key:
            return None

        # Check cache first
        cached = self.cache.get("google_nearby", lat, lng, radius)
        if cached:
            return cached

        params = {
            "location": f"{lat},{lng}",
            "radius": radius,
            "type": "restaurant",
            "key": self.api_key
        }

        async with session.get(self.nearby_search_url, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

            if data.get("status") == "OK" and data.get("results"):
                result = {
                    "name": data["results"][0].get("name"),
                    "place_id": data["results"][0].get("place_id"),
                    "address": data["results"][0].get("vicinity")
                }
                self.cache.set("google_nearby", lat, lng, radius, value=result)
                return result

        return None

    async def find_restaurant(
        self,
        session: aiohttp.ClientSession,
        llc_name: Optional[str] = None,
        address: Optional[str] = None,
        city: Optional[str] = None,
        state: Optional[str] = None,
        lat: Optional[float] = None,
        lng: Optional[float] = None
    ) -> Optional[dict]:
        """Find restaurant using best available search method.
        
        Strategy:
        1. If address info available, try Text Search (New) first - better for LLC matching
        2. If coordinates available, fall back to Nearby Search
        3. Return None if neither method succeeds
        
        Args:
            session: aiohttp client session
            llc_name: Business/LLC name (for text search)
            address: Street address (for text search)
            city: City name (for text search)
            state: State code (for text search)
            lat: Latitude (for nearby search fallback and text search location bias)
            lng: Longitude (for nearby search fallback and text search location bias)
            
        Returns:
            dict with 'name', 'place_id', 'address' or None if not found
        """
        # Try Text Search first if we have LLC name and some location info
        if llc_name and (address or city or state):
            result = await self.search_by_text(
                session, llc_name, address, city, state, lat, lng
            )
            if result and result.get("name"):
                return result

        # Fall back to Nearby Search if we have coordinates
        if lat is not None and lng is not None:
            result = await self.search_nearby(session, lat, lng)
            if result and result.get("name"):
                return result

        return None


class YelpClient:
    """Yelp Fusion API client for restaurant search and verification.
    
    Uses Yelp Business Search API to find restaurant details by name and location.
    Useful as an alternative data source for restaurant name resolution.
    
    API Documentation:
        Endpoint: https://api.yelp.com/v3/businesses/search
        Auth: Bearer token in Authorization header
        Params: term (search query), location (city, state), categories, limit
    """

    def __init__(self, api_key: str, cache: CacheManager):
        self.api_key = api_key
        self.cache = cache
        self.base_url = "https://api.yelp.com/v3/businesses/search"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search_restaurant(
        self,
        session: aiohttp.ClientSession,
        llc_name: str,
        address: str,
        city: str,
        state: str
    ) -> Optional[dict]:
        """Search for a restaurant using Yelp Business Search API.
        
        Args:
            session: aiohttp ClientSession for making requests
            llc_name: Business/LLC name to search for
            address: Street address (used to refine location)
            city: City name for location search
            state: State code for location search
            
        Returns:
            Dict with restaurant info, or None if not found:
            {
                "name": str,      # Business name from Yelp
                "address": str,   # Full address
                "phone": str      # Business phone number
            }
        """
        if not self.api_key:
            log_verbose("[Yelp] No API key configured - skipping search")
            return None

        if not llc_name or not city:
            log_verbose(f"[Yelp] Missing required params: llc_name={llc_name}, city={city}")
            return None

        # Build location string from city and state
        location = f"{city}, {state}" if state else city

        # Check cache first - normalize cache key
        cache_key = f"{llc_name.lower().strip()}:{location.lower().strip()}"
        cached = self.cache.get("yelp_search", cache_key)
        if cached:
            log_verbose(f"[Yelp] Cache hit for {llc_name}")
            return cached

        # Build search term from LLC name
        # Clean up common LLC suffixes for better matching
        search_term = llc_name
        for suffix in [" LLC", " INC", " CORP", " CO", " LTD", " L.L.C.", " INC."]:
            if search_term.upper().endswith(suffix):
                search_term = search_term[:-len(suffix)].strip()
                break

        # Build request headers with Bearer token authentication
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }

        # Build request parameters
        params = {
            "term": search_term,
            "location": location,
            "categories": "restaurants",
            "limit": 3  # Get top 3 results to find best match
        }

        log_verbose(f"[Yelp] Searching: term='{search_term}' location='{location}'")

        try:
            async with session.get(self.base_url, headers=headers, params=params) as resp:
                if resp.status == 429:
                    log_verbose("[Yelp] Rate limit exceeded")
                    raise Exception("Yelp API rate limited")  # Let retry handle it

                if resp.status == 401:
                    log_verbose("[Yelp] Authentication failed - check API key")
                    return None

                if resp.status != 200:
                    error_text = await resp.text()
                    log_verbose(f"[Yelp] API error {resp.status}: {error_text[:200]}")
                    return None

                data = await resp.json()

                # Extract businesses from response
                businesses = data.get("businesses", [])
                if not businesses:
                    log_verbose(f"[Yelp] No results found for {llc_name}")
                    return None

                # Take the first/best match
                business = businesses[0]

                # Build address from location components
                location_data = business.get("location", {})
                address_parts = []
                if location_data.get("address1"):
                    address_parts.append(location_data["address1"])
                if location_data.get("city"):
                    address_parts.append(location_data["city"])
                if location_data.get("state"):
                    address_parts.append(location_data["state"])
                if location_data.get("zip_code"):
                    address_parts.append(location_data["zip_code"])
                full_address = ", ".join(address_parts) if address_parts else ""

                # Format phone number (Yelp returns with +1 prefix)
                phone = business.get("phone", "")
                if phone and phone.startswith("+1"):
                    phone = phone[2:]  # Remove +1 prefix

                result = {
                    "name": business.get("name"),
                    "address": full_address,
                    "phone": phone
                }

                log_verbose(f"[Yelp] Found: {result['name']} at {result['address']}")

                # Cache the result
                self.cache.set("yelp_search", cache_key, value=result)
                return result

        except aiohttp.ClientError as e:
            log_verbose(f"[Yelp] Connection error for {llc_name}: {e}")
            raise  # Let retry decorator handle it
        except Exception as e:
            if "rate limited" in str(e).lower():
                raise  # Re-raise rate limit for retry
            log_verbose(f"[Yelp] Error for {llc_name}: {e}")
            return None


def clean_perplexity_text(text: str) -> str:
    """Clean Perplexity response text by removing markdown and citations."""
    if not text:
        return ""
    # Remove markdown code blocks (```json ... ``` or ``` ... ```)
    text = re.sub(r'```(?:json|JSON)?\s*', '', text)
    text = re.sub(r'```', '', text)
    # Remove markdown bold/italic
    text = re.sub(r'\*+', '', text)
    # Remove citation references like [1], [2][3], etc.
    text = re.sub(r'\[\d+\]', '', text)
    # Remove any remaining brackets with numbers
    text = re.sub(r'\s*\[[\d,\s]+\]\s*', '', text)
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()


class PerplexityClient:
    """Perplexity API client (via OpenRouter) for restaurant name resolution and owner discovery."""

    def __init__(self, api_key: str, cache: CacheManager):
        self.api_key = api_key
        self.cache = cache
        # Use OpenRouter to access Perplexity models
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        ) if api_key else None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def resolve_restaurant_name(
        self,
        llc_name: str,
        address: str,
        city: str,
        state: str,
        zip_code: str = "",
        website: str = ""
    ) -> Optional[str]:
        """Resolve the actual restaurant name (DBA) from LLC name.
        
        Uses a restaurant identification specialist prompt to find the actual
        restaurant name (DBA name) based on the provided LLC/business name and location information. Many restaurants operate under a different name than their legal LLC name.
        """
        if not self.client:
            return None

        cache_key = f"{llc_name}:{address}:{city}:{state}"
        cached = self.cache.get("perplexity_name", cache_key)
        if cached:
            return cached.get("name")

        # System prompt - restaurant identification specialist persona
        system_prompt = """You are a restaurant identification specialist. Your task is to find the actual restaurant or bar name (DBA name) that operates at the EXACT address provided. Many restaurants/bars operate under a different name than their legal LLC name.

CRITICAL RULES:
- ONLY return the name of a restaurant/bar/establishment that is ACTUALLY LOCATED at the exact address provided
- Do NOT guess or return nearby businesses - the address must match exactly
- Search for what business is currently operating at that specific street address
- If you cannot find a restaurant/bar at that exact address, return "UNKNOWN"
- Return only the business name as a plain string (no explanations)
- Use proper capitalization as it would appear on signage
- Include restaurants, bars, nightclubs, taverns, and similar food/drink establishments"""

        # User prompt with location context
        user_prompt = f"""What restaurant, bar, or food/drink establishment operates at this EXACT address?

LLC/Business Name: {llc_name}
Street Address: {address}
City: {city}
State: {state}
Zip Code: {zip_code if zip_code else 'Unknown'}

Search for the actual business name that customers would see at {address}, {city}, {state}. Return ONLY the business name, or "UNKNOWN" if you cannot confirm a restaurant/bar at this exact address."""

        try:
            response = await self.client.chat.completions.create(
                model="perplexity/sonar-pro",  # OpenRouter model path
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=50  # Reduced to encourage concise response
            )
            name = response.choices[0].message.content.strip()
            name = clean_perplexity_text(name)  # Remove markdown/citations

            # Extract just the restaurant name (take first line, before any explanation)
            name = name.split('\n')[0].strip()
            # If response contains explanation after the name, extract just the name
            # Look for patterns like "Name - explanation" or "Name. explanation"
            if ' - ' in name:
                name = name.split(' - ')[0].strip()
            if '. ' in name and len(name.split('. ')[0]) < 50:
                name = name.split('. ')[0].strip()
            # Also check for "This is" which indicates explanation started
            if ' This is ' in name:
                name = name.split(' This is ')[0].strip()
            if ' is the ' in name:
                name = name.split(' is the ')[0].strip()

            # Validate: reasonable length for a restaurant name (2-50 chars)
            if name and 2 <= len(name) <= 50 and name.upper() != "UNKNOWN":
                self.cache.set("perplexity_name", cache_key, value={"name": name})
                return name

            log_verbose(f"  Perplexity returned invalid name: '{name[:100]}...'")
        except Exception as e:
            print(f"Perplexity error resolving name for {llc_name}: {e}")

        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def find_owners(
        self,
        restaurant_name: str,
        llc_name: str,
        address: str,
        city: str,
        state: str,
        website: str = ""
    ) -> list[str]:
        """Find restaurant owners using web search.
        
        Uses a restaurant ownership research assistant prompt to find current
        owners. Returns results as a list of owner names parsed from JSON array.
        """
        if not self.client:
            return []

        cache_key = f"{restaurant_name}:{llc_name}:{city}:{state}"
        cached = self.cache.get("perplexity_owners", cache_key)
        if cached:
            return cached.get("owners", [])

        # System prompt - ownership research assistant persona
        system_prompt = """You are a restaurant ownership research assistant. Your task is to find the actual owner(s) of the specified restaurant/bar/establishment.

CRITICAL RULES:
- Search for OWNERS, FOUNDERS, or PROPRIETORS - people who have ownership stake
- Do NOT include: managers, general managers, executive chefs (unless also owner), employees, registered agents
- Look for ownership information in news articles, press releases, social media, business filings
- Return names as a JSON array: ["First Last", "First Last"]
- If no confirmed owners found, return: []
- Use full names (first and last) when available
- Only include names you are confident are actual owners"""

        # User prompt with full context
        user_prompt = f"""Who OWNS this restaurant/bar? Find the actual owner(s).

Restaurant/Bar Name: {restaurant_name}
LLC Name: {llc_name}
Location: {address}, {city}, {state}
Website: {website if website else 'None'}

Search for the owner(s) or founder(s) of {restaurant_name} in {city}, {state}. Return a JSON array of owner names, or [] if unknown.

Example response formats:
["John Smith"]
["John Smith", "Jane Doe"]
[]"""

        try:
            response = await self.client.chat.completions.create(
                model="perplexity/sonar-pro",  # OpenRouter model path
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            content = clean_perplexity_text(content)  # Remove markdown/citations

            # Parse JSON array response
            owners = self._parse_owners_response(content)
            
            self.cache.set("perplexity_owners", cache_key, value={"owners": owners})
            return owners

        except Exception as e:
            print(f"Perplexity error finding owners for {restaurant_name}: {e}")

        return []

    async def find_owners_multi_strategy(
        self,
        restaurant_name: str,
        llc_name: str,
        address: str,
        city: str,
        state: str,
        website: str = "",
        persons_from_csv: list["PersonInfo"] = None
    ) -> list["OwnerResult"]:
        """Find restaurant owners using multiple search strategies.
        
        Tries three approaches:
        1. Primary structured query (who owns the restaurant)
        2. Founder query (who founded the restaurant)
        3. LLC lookup (principals/members of the LLC)
        
        Returns combined deduplicated results with confidence scores.
        CSV matches are prioritized with higher confidence.
        """
        if persons_from_csv is None:
            persons_from_csv = []
            
        all_results: list[OwnerResult] = []
        seen_names: set[str] = set()
        
        # Strategy 1: Primary ownership query (existing approach)
        primary_owners = await self._find_owners_primary(
            restaurant_name, llc_name, address, city, state, website
        )
        for name in primary_owners:
            normalized = name.lower().strip()
            if normalized not in seen_names:
                seen_names.add(normalized)
                all_results.append(OwnerResult(
                    name=name,
                    confidence=0.6,
                    strategy="primary"
                ))
        
        # Strategy 2: Founder query
        founder_owners = await self._find_owners_founder(
            restaurant_name, city, state
        )
        for name in founder_owners:
            normalized = name.lower().strip()
            if normalized not in seen_names:
                seen_names.add(normalized)
                all_results.append(OwnerResult(
                    name=name,
                    confidence=0.4,
                    strategy="founder"
                ))
        
        # Strategy 3: LLC lookup
        if llc_name and llc_name != restaurant_name:
            llc_owners = await self._find_owners_llc(llc_name, state)
            for name in llc_owners:
                normalized = name.lower().strip()
                if normalized not in seen_names:
                    seen_names.add(normalized)
                    all_results.append(OwnerResult(
                        name=name,
                        confidence=0.4,
                        strategy="llc_lookup"
                    ))
        
        # Boost confidence for CSV matches
        for result in all_results:
            matched_person = fuzzy_match_owner(result.name, persons_from_csv)
            if matched_person:
                result.matched_csv_person = matched_person
                if matched_person.phone:
                    result.confidence = 1.0
                    result.strategy = "csv_match_with_phone"
                else:
                    result.confidence = 0.8
                    result.strategy = "csv_match_no_phone"
        
        # Sort by confidence descending
        all_results.sort(key=lambda x: x.confidence, reverse=True)
        
        log_verbose(f"  Multi-strategy found {len(all_results)} unique owners")
        for r in all_results:
            log_verbose(f"    - {r}")
        
        return all_results

    async def _find_owners_primary(
        self,
        restaurant_name: str,
        llc_name: str,
        address: str,
        city: str,
        state: str,
        website: str = ""
    ) -> list[str]:
        """Primary strategy: Who owns this restaurant? (existing logic)"""
        # This is essentially the existing find_owners() logic
        if not self.client:
            return []

        cache_key = f"primary:{restaurant_name}:{llc_name}:{city}:{state}"
        cached = self.cache.get("perplexity_owners_multi", cache_key)
        if cached:
            return cached.get("owners", [])

        system_prompt = """You are a restaurant ownership research assistant. Your task is to find the actual owner(s) of the specified restaurant/bar/establishment.

CRITICAL RULES:
- Search for OWNERS, FOUNDERS, or PROPRIETORS - people who have ownership stake
- Do NOT include: managers, general managers, executive chefs (unless also owner), employees, registered agents
- Look for ownership information in news articles, press releases, social media, business filings
- Return names as a JSON array: ["First Last", "First Last"]
- If no confirmed owners found, return: []
- Use full names (first and last) when available
- Only include names you are confident are actual owners"""

        user_prompt = f"""Who OWNS this restaurant/bar? Find the actual owner(s).

Restaurant/Bar Name: {restaurant_name}
LLC Name: {llc_name}
Location: {address}, {city}, {state}
Website: {website if website else 'None'}

Search for the owner(s) or founder(s) of {restaurant_name} in {city}, {state}. Return a JSON array of owner names, or [] if unknown.

Example response formats:
["John Smith"]
["John Smith", "Jane Doe"]
[]"""

        try:
            response = await self.client.chat.completions.create(
                model="perplexity/sonar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            content = clean_perplexity_text(content)
            owners = self._parse_owners_response(content)
            self.cache.set("perplexity_owners_multi", cache_key, value={"owners": owners})
            return owners
        except Exception as e:
            print(f"Perplexity error (primary strategy) for {restaurant_name}: {e}")
        return []

    async def _find_owners_founder(
        self,
        restaurant_name: str,
        city: str,
        state: str
    ) -> list[str]:
        """Alternative strategy: Who founded this restaurant?"""
        if not self.client:
            return []

        cache_key = f"founder:{restaurant_name}:{city}:{state}"
        cached = self.cache.get("perplexity_owners_multi", cache_key)
        if cached:
            return cached.get("owners", [])

        system_prompt = """You are a restaurant research assistant. Your task is to find who FOUNDED or STARTED a restaurant.

CRITICAL RULES:
- Focus on founders, co-founders, and original creators of the restaurant
- Founders are often still owners or were the original owners
- Return names as a JSON array: ["First Last", "First Last"]
- If no confirmed founders found, return: []
- Use full names (first and last) when available"""

        user_prompt = f"""Who FOUNDED {restaurant_name} in {city}, {state}?

Search for news articles, interviews, or press about who started or created this restaurant. Return a JSON array of founder names, or [] if unknown.

Example response formats:
["John Smith"]
["John Smith", "Jane Doe"]
[]"""

        try:
            response = await self.client.chat.completions.create(
                model="perplexity/sonar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            content = clean_perplexity_text(content)
            owners = self._parse_owners_response(content)
            self.cache.set("perplexity_owners_multi", cache_key, value={"owners": owners})
            return owners
        except Exception as e:
            log_verbose(f"  Perplexity error (founder strategy) for {restaurant_name}: {e}")
        return []

    async def _find_owners_llc(
        self,
        llc_name: str,
        state: str
    ) -> list[str]:
        """LLC lookup strategy: Who are the principals/members of the LLC?"""
        if not self.client:
            return []

        cache_key = f"llc:{llc_name}:{state}"
        cached = self.cache.get("perplexity_owners_multi", cache_key)
        if cached:
            return cached.get("owners", [])

        system_prompt = """You are a business records research assistant. Your task is to find the principals, members, or managing members of an LLC.

CRITICAL RULES:
- Search for LLC members, principals, managing members, or registered agents who are also owners
- Look for state business filing records, annual reports, or news mentioning LLC principals
- Do NOT include: corporate registered agents (like CT Corporation), law firms, or accounting firms
- Return names as a JSON array: ["First Last", "First Last"]
- If no confirmed principals found, return: []
- Use full names (first and last) when available"""

        user_prompt = f"""Who are the principals or members of {llc_name} in {state}?

Search for business filings, state records, or news about the members/principals of this LLC. Return a JSON array of individual names (not company names), or [] if unknown.

Example response formats:
["John Smith"]
["John Smith", "Jane Doe"]
[]"""

        try:
            response = await self.client.chat.completions.create(
                model="perplexity/sonar-pro",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200
            )
            content = response.choices[0].message.content.strip()
            content = clean_perplexity_text(content)
            owners = self._parse_owners_response(content)
            self.cache.set("perplexity_owners_multi", cache_key, value={"owners": owners})
            return owners
        except Exception as e:
            log_verbose(f"  Perplexity error (LLC strategy) for {llc_name}: {e}")
        return []

    def _parse_owners_response(self, content: str) -> list[str]:
        """Parse owner names from response, handling JSON array or text fallback.
        
        Args:
            content: Response content from Perplexity API
            
        Returns:
            List of cleaned owner names
        """
        if not content or content.upper() == "UNKNOWN":
            return []
        
        owners = []
        
        # Strategy 1: Try direct JSON parsing
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for name in parsed:
                    if isinstance(name, str) and self._is_valid_owner_name(name):
                        owners.append(name.strip())
                if owners:
                    return owners
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract JSON array from text using regex
        json_match = re.search(r'\[([^\]]*)\]', content)
        if json_match:
            try:
                json_str = '[' + json_match.group(1) + ']'
                parsed = json.loads(json_str)
                if isinstance(parsed, list):
                    for name in parsed:
                        if isinstance(name, str) and self._is_valid_owner_name(name):
                            owners.append(name.strip())
                    if owners:
                        return owners
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Fallback to text parsing (legacy behavior)
        # Remove bullets/numbers
        content = re.sub(r'^[\d\.\-\*\•]+\s*', '', content, flags=re.MULTILINE)
        
        # Split by common delimiters
        for line in content.split("\n"):
            line = line.strip()
            line = re.sub(r'\s*[\(\[].*?[\)\]]', '', line)  # Remove parenthetical info
            
            if self._is_valid_owner_name(line):
                owners.append(line)
                # For fallback parsing, only take first valid name
                break
        
        return owners
    
    def _is_valid_owner_name(self, name: str) -> bool:
        """Validate that a string looks like a valid owner name."""
        if not name:
            return False
        name = name.strip()
        # Basic validation: reasonable length and not a keyword
        if len(name) < 3 or len(name) > 60:
            return False
        # Reject common invalid responses
        invalid_patterns = ["UNKNOWN", "N/A", "NONE", "NULL", "[]", "```", "JSON", "ARRAY"]
        if name.upper() in invalid_patterns or any(p in name.upper() for p in ["```", "JSON"]):
            return False
        # Should contain at least one space (first + last name)
        if " " not in name:
            return False
        # Should contain mostly letters (names shouldn't have lots of special chars)
        letter_count = sum(1 for c in name if c.isalpha())
        if letter_count < len(name) * 0.6:  # At least 60% letters
            return False
        return True


class WhitepagesClient:
    """Whitepages Pro API client for owner personal information lookup.

    Uses Whitepages Pro 2.2 Person API to find owner's personal contact information
    (home address, personal phone) distinct from restaurant contact info.

    API Documentation:
        Endpoint: https://proapi.whitepages.com/2.2/person.json
        Auth: api_key query parameter
        Params: api_key, name, city, state_code (2-letter state code)
        Response: {"results": [...]} with locations[], phones[] arrays
    """

    def __init__(self, api_key: str, cache: CacheManager):
        self.api_key = api_key
        self.cache = cache
        self.base_url = "https://proapi.whitepages.com/2.2/person.json"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def lookup_person(
        self,
        session: aiohttp.ClientSession,
        name: str,
        city: str,
        state: str
    ) -> Optional[dict]:
        """Look up a person's personal contact info by name and location.

        Args:
            session: aiohttp ClientSession for making requests
            name: Full name of the person to look up
            city: City where the person is located
            state: 2-letter state code (e.g., "SC", "CA")

        Returns:
            Dict with personal contact info, or None if not found:
            {
                "personal_address": str,  # Street address
                "personal_city": str,
                "personal_state": str,
                "personal_zip": str,
                "personal_phone": str
            }
        """
        if not self.api_key:
            print("[Whitepages] No API key configured - skipping lookup")
            return None

        if not name or not city or not state:
            print(f"[Whitepages] Missing required params: name={name}, city={city}, state={state}")
            return None

        # Check cache first - use normalized cache key
        cache_key = f"{name.lower().strip()}:{city.lower().strip()}:{state.upper().strip()}"
        cached = self.cache.get("whitepages_person", cache_key)
        if cached:
            print(f"[Whitepages] Cache hit for {name}")
            return cached

        # Authentication via query parameter (NOT header)
        params = {
            "api_key": self.api_key,
            "name": name,
            "city": city,
            "state_code": state.upper()[:2]  # API expects 2-letter state code
        }

        print(f"[Whitepages] Looking up: {name} in {city}, {state}")

        try:
            async with session.get(self.base_url, params=params) as resp:
                if resp.status == 429:
                    print(f"[Whitepages] Rate limit exceeded")
                    return None
                if resp.status == 401:
                    print(f"[Whitepages] Authentication failed - check API key")
                    return None
                if resp.status != 200:
                    text = await resp.text()
                    print(f"[Whitepages] API error {resp.status}: {text[:200]}")
                    return None

                data = await resp.json()
                print(f"[Whitepages] Response received, parsing results...")

                # Response structure: {"results": [...]}
                results = data.get("results", [])
                if not results:
                    print(f"[Whitepages] No results found for {name}")
                    return None

                # Take first/best match
                person = results[0]

                # Initialize result structure (no email - API doesn't return it)
                result = {
                    "personal_address": None,
                    "personal_city": None,
                    "personal_state": None,
                    "personal_zip": None,
                    "personal_phone": None
                }

                # Parse location (first one)
                locations = person.get("locations", [])
                if locations:
                    loc = locations[0]
                    result["personal_address"] = loc.get("standard_address_line1", "")
                    result["personal_city"] = loc.get("city", "")
                    result["personal_state"] = loc.get("state_code", "")
                    result["personal_zip"] = loc.get("postal_code", "")
                    print(f"[Whitepages] Found address: {result['personal_address']}, {result['personal_city']}, {result['personal_state']} {result['personal_zip']}")

                # Parse phone (first one)
                phones = person.get("phones", [])
                if phones:
                    result["personal_phone"] = phones[0].get("phone_number", "")
                    print(f"[Whitepages] Found phone: {result['personal_phone']}")

                # Cache the result
                self.cache.set("whitepages_person", cache_key, value=result)
                print(f"[Whitepages] Cached result for {name}")
                return result

        except aiohttp.ClientError as e:
            print(f"[Whitepages] Connection error for {name}: {e}")
            return None
        except Exception as e:
            print(f"[Whitepages] Error for {name}: {e}")
            return None


# ============================================================================
# Data Processing
# ============================================================================

def clean_str(val) -> str:
    """Clean a value to string, handling NaN and None."""
    if pd.isna(val) or val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def clean_fein(val) -> str:
    """Clean FEIN value, removing decimal places from float conversion."""
    if pd.isna(val) or val is None:
        return ""
    # If it's a float that represents an int, convert properly
    if isinstance(val, float) and val == int(val):
        return str(int(val))
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    # Remove .0 suffix if present
    if s.endswith(".0"):
        s = s[:-2]
    return s


def parse_csv_row(row: pd.Series) -> RestaurantRecord:
    """Parse a CSV row into a RestaurantRecord."""
    record = RestaurantRecord(
        fein=clean_fein(row.get("fein")),
        llc_name=clean_str(row.get("name")),
        lat=float(row["lat"]) if pd.notna(row.get("lat")) and row.get("lat") != "" else None,
        lng=float(row["long"]) if pd.notna(row.get("long")) and row.get("long") != "" else None,
        address=clean_str(row.get("address")),
        city=clean_str(row.get("city")),
        state=clean_str(row.get("state")),
        zip_code=clean_str(row.get("zip")),
        phone=clean_str(row.get("phone")),
        email=clean_str(row.get("email1")),
        county=clean_str(row.get("county")),
        expdate=clean_str(row.get("expdate")),
        website=clean_str(row.get("website"))
    )

    # Parse persons from CSV (name1-10, phone1-10)
    for i in range(1, 11):
        name_col = f"name{i}"
        phone_col = f"phone{i}"

        name = clean_str(row.get(name_col))
        phone = clean_str(row.get(phone_col))

        if name:
            record.persons_from_csv.append(PersonInfo(name=name, phone=phone if phone else None, source="csv"))

    return record


def extract_dba_from_name(llc_name: str) -> Optional[str]:
    """Extract DBA from LLC name if present (e.g., 'BUMPER CROP LLC DBA FIG')."""
    dba_match = re.search(r'\bDBA\s+(.+)$', llc_name, re.IGNORECASE)
    if dba_match:
        return dba_match.group(1).strip()
    return None


def fuzzy_match_owner(
    owner_name: str,
    persons: list[PersonInfo],
    threshold: int = 80
) -> Optional[PersonInfo]:
    """Find a matching person using fuzzy string matching."""
    owner_name_normalized = owner_name.lower().strip()

    best_match = None
    best_score = 0

    for person in persons:
        person_name_normalized = person.name.lower().strip()

        # Try different matching strategies
        scores = [
            fuzz.ratio(owner_name_normalized, person_name_normalized),
            fuzz.partial_ratio(owner_name_normalized, person_name_normalized),
            fuzz.token_sort_ratio(owner_name_normalized, person_name_normalized)
        ]

        max_score = max(scores)
        if max_score > best_score and max_score >= threshold:
            best_score = max_score
            best_match = person

    return best_match


async def process_record(
    record: RestaurantRecord,
    session: aiohttp.ClientSession,
    google_client: GooglePlacesClient,
    perplexity_client: PerplexityClient,
    whitepages_client: WhitepagesClient,
    yelp_client: Optional[YelpClient],
    semaphore: asyncio.Semaphore
) -> RestaurantRecord:
    """Process a single restaurant record through the enrichment pipeline."""

    async with semaphore:
        log_verbose(f"Processing: {record.llc_name}")

        # Step 1: Resolve restaurant name
        # First check if DBA is in the LLC name
        dba_name = extract_dba_from_name(record.llc_name)

        if dba_name:
            record.restaurant_name = dba_name
            log_verbose(f"  DBA extracted from LLC: '{dba_name}'")
        else:
            # Try Google Places - now supports both Text Search and Nearby Search
            # Text Search works with LLC name + address, Nearby Search needs coordinates
            has_address_info = record.address or record.city or record.state
            has_coordinates = record.lat and record.lng

            if has_address_info or has_coordinates:
                log_verbose(f"  Trying Google Places (addr={has_address_info}, coords={has_coordinates})")
                place = await google_client.find_restaurant(
                    session,
                    llc_name=record.llc_name,
                    address=record.address,
                    city=record.city,
                    state=record.state,
                    lat=record.lat,
                    lng=record.lng
                )
                if place and place.get("name"):
                    record.restaurant_name = place["name"]
                    log_verbose(f"  Google Places found: '{place['name']}'")
                else:
                    log_verbose(f"  Google Places: no result")

        # Try Yelp as second fallback (if configured)
        if not record.restaurant_name and yelp_client:
            log_verbose(f"  Trying Yelp for name resolution...")
            try:
                yelp_result = await yelp_client.search_restaurant(
                    session,
                    llc_name=record.llc_name,
                    address=record.address,
                    city=record.city,
                    state=record.state
                )
                if yelp_result and yelp_result.get("name"):
                    record.restaurant_name = yelp_result["name"]
                    log_verbose(f"  Yelp found: '{yelp_result['name']}'")
                else:
                    log_verbose(f"  Yelp: no result")
            except Exception as e:
                log_verbose(f"  Yelp error: {e}")

        # Fall back to Perplexity if still no name
        if not record.restaurant_name:
            log_verbose(f"  Trying Perplexity for name resolution...")
            resolved_name = await perplexity_client.resolve_restaurant_name(
                record.llc_name,
                record.address,
                record.city,
                record.state,
                record.zip_code,
                record.website
            )
            if resolved_name:
                record.restaurant_name = resolved_name
                log_verbose(f"  Perplexity resolved name: '{resolved_name}'")
            else:
                log_verbose(f"  Perplexity: no name resolved")

        # Default to LLC name if nothing found
        if not record.restaurant_name:
            record.restaurant_name = record.llc_name
            log_verbose(f"  Using LLC name as fallback: '{record.llc_name}'")

        log_verbose(f"  Final restaurant name: '{record.restaurant_name}'")

        # Step 2: Find owners via multi-strategy Perplexity search
        log_verbose(f"  Finding owners for '{record.restaurant_name}' (multi-strategy)...")
        owner_results = await perplexity_client.find_owners_multi_strategy(
            record.restaurant_name,
            record.llc_name,
            record.address,
            record.city,
            record.state,
            record.website,
            record.persons_from_csv
        )

        # Step 3: Select best owner based on confidence scores
        best_owner = None
        best_owner_result = None

        if owner_results:
            # Results are already sorted by confidence descending
            best_owner_result = owner_results[0]
            
            log_verbose(f"  Best owner result: {best_owner_result}")
            
            # Create PersonInfo from the best result
            if best_owner_result.matched_csv_person:
                # Use CSV-matched person's contact info
                matched = best_owner_result.matched_csv_person
                best_owner = PersonInfo(
                    name=best_owner_result.name,
                    phone=matched.phone,
                    email=record.email if record.email and matched.name.lower() in record.email.lower() else matched.email,
                    source=f"csv ({best_owner_result.strategy})"
                )
            else:
                # No CSV match, use Perplexity-discovered owner
                best_owner = PersonInfo(
                    name=best_owner_result.name,
                    source=f"perplexity ({best_owner_result.strategy})"
                )

        # Fallback: If no owner from Perplexity, use first person from CSV with phone
        if not best_owner and record.persons_from_csv:
            for person in record.persons_from_csv:
                if person.name and person.phone:
                    best_owner = person
                    log_verbose(f"  Using CSV person as fallback: '{person.name}'")
                    break
            # If still none, just take first person
            if not best_owner and record.persons_from_csv[0].name:
                best_owner = record.persons_from_csv[0]
                log_verbose(f"  Using first CSV person: '{best_owner.name}'")

        if best_owner:
            record.owners.append(best_owner)
            confidence_str = f", confidence={best_owner_result.confidence:.2f}" if best_owner_result else ""
            log_verbose(f"  Final owner: '{best_owner.name}' (source={best_owner.source}{confidence_str})")
        else:
            log_verbose(f"  No owner found")

        # Step 4: Enrich owner with Whitepages personal info
        if best_owner and record.city and record.state:
            wp_result = await whitepages_client.lookup_person(
                session,
                best_owner.name,
                record.city,
                record.state
            )
            if wp_result:
                best_owner.personal_address = wp_result.get("personal_address")
                best_owner.personal_city = wp_result.get("personal_city")
                best_owner.personal_state = wp_result.get("personal_state")
                best_owner.personal_zip = wp_result.get("personal_zip")
                best_owner.personal_phone = wp_result.get("personal_phone")
                # Update source to indicate Whitepages enrichment
                if wp_result.get("personal_phone"):
                    best_owner.source = "whitepages"

        return record


def format_output_row(record: RestaurantRecord) -> dict:
    """Format a RestaurantRecord into a single output row.

    Output format prioritizes owner's personal contact info from Whitepages.
    Address/City/State/Zip are the owner's home address (where they live),
    not the restaurant address.
    """
    owner = record.owners[0] if record.owners else None

    # Build owner's personal address as combined string: "street, city, state zip"
    # Prioritize Whitepages personal address, fall back to restaurant address
    if owner and owner.personal_address:
        # Use Whitepages personal address
        addr_street = owner.personal_address
        addr_city = owner.personal_city or ""
        addr_state = owner.personal_state or ""
        addr_zip = owner.personal_zip or ""
        # Format as combined address string
        address_combined = f"{addr_street}, {addr_city}, {addr_state} {addr_zip}".strip()
    else:
        # Fall back to restaurant address
        addr_street = record.address
        addr_city = record.city
        addr_state = record.state
        addr_zip = record.zip_code
        address_combined = f"{addr_street}, {addr_city}, {addr_state} {addr_zip}".strip() if addr_street else ""

    # Phone/Email: prioritize Whitepages personal, then CSV match, then restaurant
    if owner and owner.personal_phone:
        phone = owner.personal_phone
    elif owner and owner.phone:
        phone = owner.phone
    else:
        phone = record.phone

    # Email fallback - Whitepages doesn't return email, use existing data if available
    if owner and owner.email:
        email = owner.email
    else:
        email = record.email

    return {
        "FEIN": record.fein,
        "Name": record.restaurant_name,
        "OwnerName": owner.name if owner else "",
        "Address": address_combined,
        "City": addr_city,
        "State": addr_state,
        "Zip": addr_zip,
        "Phone": phone,
        "Email": email,
        "County": record.county,
        "Expdate": record.expdate,
        "Website": record.website,
        "LLC_Name": record.llc_name,
        "ContactSource": owner.source if owner else "",
    }


# ============================================================================
# Main Pipeline
# ============================================================================

async def process_batch(
    records: list[RestaurantRecord],
    config: Config,
    batch_size: int = 10
) -> list[RestaurantRecord]:
    """Process records in parallel batches."""

    cache = CacheManager(config.cache_dir)
    google_client = GooglePlacesClient(config.google_places_api_key, cache)
    perplexity_client = PerplexityClient(config.openrouter_api_key, cache)
    whitepages_client = WhitepagesClient(config.whitepages_api_key, cache)
    
    # Optional data source clients (only used if API keys are configured)
    yelp_client = YelpClient(config.yelp_api_key, cache) if config.yelp_api_key else None

    # Semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(batch_size)

    async with aiohttp.ClientSession() as session:
        tasks = [
            process_record(
                record,
                session,
                google_client,
                perplexity_client,
                whitepages_client,
                yelp_client,
                semaphore
            )
            for record in records
        ]

        # Process with progress bar
        results = await tqdm_asyncio.gather(*tasks, desc="Processing records")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Restaurant Lead Enrichment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.csv -o output.csv
  python main.py input.csv -o output.csv --batch-size 20
  python main.py input.xlsx -o output.csv

Environment Variables:
  GOOGLE_PLACES_API_KEY  - Google Places API key
  OPENROUTER_API_KEY     - OpenRouter API key (for Perplexity sonar-pro)
        """
    )

    parser.add_argument("input", help="Input CSV or Excel file")
    parser.add_argument("-o", "--output", default="output.csv", help="Output CSV file")
    parser.add_argument("--batch-size", type=int, default=10, help="Parallel batch size")
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of records to process (for testing)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging for debugging")

    args = parser.parse_args()

    # Set verbose mode
    global VERBOSE
    VERBOSE = args.verbose

    # Initialize config
    config = Config(cache_dir=Path(args.cache_dir))
    config.validate()

    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    print(f"Reading input file: {args.input}")

    if input_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(input_path)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} records")

    # Parse records
    print("Parsing records...")
    records = [parse_csv_row(row) for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing")]

    # Apply limit if specified
    if args.limit is not None:
        records = records[:args.limit]
        print(f"Limited to {len(records)} records")

    # Process records
    print(f"Processing records with batch size {args.batch_size}...")
    processed_records = asyncio.run(process_batch(records, config, args.batch_size))

    # Format output
    print("Formatting output...")
    output_rows = [format_output_row(record) for record in processed_records]

    # Write output
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(args.output, index=False)

    print(f"\nOutput written to: {args.output}")
    print(f"Total output rows: {len(output_rows)}")

    # Summary
    owners_found = sum(1 for r in processed_records if r.owners)
    owners_with_phone = sum(1 for r in processed_records for o in r.owners if o.phone)

    print(f"\nSummary:")
    print(f"  Records processed: {len(processed_records)}")
    print(f"  Records with owners found: {owners_found}")
    print(f"  Owners with phone numbers: {owners_with_phone}")


if __name__ == "__main__":
    main()
