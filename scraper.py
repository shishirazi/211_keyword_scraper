#!/usr/bin/env python3
"""
211 API Scraper - San Diego County Services

Fetches all services from the 211 API using keyword expansion
to work around the API's pagination limitation (max 10 results per query).

Usage:
    python scraper.py

Requirements:
    - API_211_KEY environment variable (or .env file)
    - pip install requests pandas python-dotenv tqdm
"""

import os
import json
import time
import hashlib
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "https://api.211.org/resources/v2"
SEARCH_ENDPOINT = f"{API_BASE_URL}/search/keyword"
DETAIL_ENDPOINT = f"{API_BASE_URL}/query/service-at-location-details"

OUTPUT_DIR = Path("output")
CACHE_DIR = Path("cache")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Load API key
API_KEY = os.environ.get("API_211_KEY")
if not API_KEY:
    raise ValueError("API_211_KEY environment variable not set. Create a .env file with API_211_KEY=your_key")


class APIClient:
    """HTTP client with retry logic and rate-limit handling."""

    def __init__(self, api_key: str, max_retries: int = 3, base_delay: float = 1.0):
        self.api_key = api_key
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.session = requests.Session()
        self.session.headers.update({
            "Api-Key": api_key,
            "Cache-Control": "no-cache",
        })

    def get(self, url: str, params: Optional[dict] = None, extra_headers: Optional[dict] = None) -> requests.Response:
        headers = extra_headers or {}

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=30)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.base_delay * (2 ** attempt)))
                    print(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response

            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2 ** attempt)
                print(f"Request failed: {e}. Retrying in {delay}s...")
                time.sleep(delay)

        raise RuntimeError("Max retries exceeded")


class ResponseCache:
    """File-based cache for API responses."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.json"

    def get(self, key: str) -> Optional[dict]:
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            with open(cache_path, "r") as f:
                return json.load(f)
        return None

    def set(self, key: str, data: dict) -> None:
        cache_path = self._get_cache_path(key)
        with open(cache_path, "w") as f:
            json.dump(data, f)


# EXHAUSTIVE keyword list to capture all services
SEARCH_KEYWORDS = [
    # Basic needs - food
    "food", "meals", "groceries", "pantry", "nutrition", "hunger", "feeding",
    "breakfast", "lunch", "dinner", "snack", "produce", "fresh", "canned",
    "soup kitchen", "food bank", "food pantry", "meal program", "congregate",
    "home delivered", "meals on wheels", "senior meals",

    # Basic needs - housing
    "housing", "shelter", "rent", "homeless", "eviction", "apartment", "home",
    "transitional", "permanent", "supportive", "emergency shelter", "motel voucher",
    "rental assistance", "deposit", "first month", "last month", "back rent",
    "section 8", "voucher", "subsidized", "affordable", "public housing",
    "rapid rehousing", "coordinated entry", "CES", "HMIS", "continuum of care",
    "housing authority", "HUD", "PSH",

    # Basic needs - utilities
    "utility", "electric", "gas", "water", "energy", "power", "bill",
    "LIHEAP", "weatherization", "solar", "disconnect", "shutoff",

    # Basic needs - material goods
    "clothing", "furniture", "household", "appliance", "mattress", "bed",
    "diapers", "formula", "baby supplies", "school supplies", "backpack",
    "thrift", "donation", "free stuff", "giveaway",

    # Health - general
    "health", "medical", "clinic", "hospital", "doctor", "nurse", "physician",
    "primary care", "urgent care", "emergency room", "ER",
    "community health", "FQHC", "federally qualified",
    "sliding scale", "free clinic", "low cost",

    # Health - mental health
    "mental health", "counseling", "therapy", "psychiatric", "psychology",
    "behavioral health", "psychologist", "social worker", "therapist",
    "depression", "anxiety", "PTSD", "trauma", "grief",
    "inpatient", "outpatient", "residential treatment",

    # Health - dental/vision
    "dental", "dentist", "teeth", "oral health",
    "vision", "eye", "glasses", "optometry", "ophthalmology",

    # Health - substance abuse
    "substance abuse", "addiction", "recovery", "detox", "sober", "sobriety",
    "AA", "NA", "alcoholics anonymous", "narcotics anonymous",
    "drug", "alcohol", "opioid", "meth", "heroin", "fentanyl",
    "MAT", "medication assisted", "methadone", "suboxone", "vivitrol",
    "IOP", "sober living", "recovery home", "halfway house",

    # Health - medications
    "prescription", "medication", "pharmacy", "medicine", "rx",
    "prescription assistance", "PAP", "patient assistance",

    # Health - specific conditions
    "HIV", "AIDS", "STD", "STI", "hepatitis", "hep c",
    "cancer", "diabetes", "heart", "asthma", "chronic", "COPD",
    "obesity", "weight", "bariatric",

    # Health - women's health
    "women's health", "reproductive", "prenatal", "prenatal care", "maternal",
    "pregnancy", "pregnant", "OB", "obstetrics", "gynecology",
    "family planning", "birth control", "contraception",
    "breastfeeding", "lactation", "WIC",

    # Health - preventive
    "immunization", "vaccine", "vaccination", "flu shot", "COVID",
    "testing", "screening", "checkup", "physical", "wellness",

    # Health - specialized
    "durable medical equipment", "DME", "prosthetics", "orthotics",
    "wheelchair", "walker", "cane", "oxygen",
    "speech therapy", "occupational therapy", "OT", "physical therapy", "PT",
    "rehabilitation", "rehab",
    "hearing", "hearing aid", "audiologist", "deaf",
    "hospice", "palliative", "end of life",
    "home health", "home care", "in-home", "caregiver",

    # Employment
    "employment", "job", "jobs", "career", "workforce", "work",
    "hire", "hiring", "employer", "employee",
    "job training", "vocational", "apprentice", "certificate",
    "resume", "interview", "job search", "job placement",
    "unemployment", "EDD", "UI",
    "career center", "one stop", "America's Job Center", "AJCC",
    "CalJOBS", "workforce development",

    # Education
    "education", "school", "learning", "class", "classes",
    "literacy", "reading", "writing", "math",
    "GED", "high school equivalency", "diploma",
    "ESL", "English", "ESOL", "English class",
    "adult education", "adult school", "continuing education",
    "college", "university", "community college", "higher education",
    "scholarship", "FAFSA", "financial aid", "Pell grant",
    "tutoring", "tutor", "homework help",

    # Children & Youth
    "child", "children", "kid", "kids", "minor",
    "youth", "teen", "teenager", "adolescent", "young adult",
    "childcare", "child care", "daycare", "day care", "babysitting",
    "preschool", "pre-k", "kindergarten", "head start", "early head start",
    "afterschool", "after school", "before school", "latchkey",
    "summer program", "summer camp", "camp", "recreation",
    "boys and girls club", "YMCA", "YWCA",
    "foster", "foster care", "foster youth", "emancipated",
    "TAY", "transition age youth", "aging out",
    "runaway", "homeless youth", "unaccompanied",
    "mentoring", "mentor", "big brothers", "big sisters",

    # Family
    "family", "families", "parent", "parenting", "parents",
    "mother", "mom", "father", "dad", "single parent",
    "expecting", "baby", "infant", "newborn",
    "adoption", "custody", "visitation", "co-parenting",
    "family counseling", "family therapy", "marriage counseling",

    # Seniors
    "senior", "seniors", "elderly", "older adult", "aging",
    "Medicare", "retirement", "retired", "retiree",
    "IHSS", "in home supportive services",
    "adult day", "adult day care", "adult day health",
    "assisted living", "nursing home", "skilled nursing", "SNF",
    "Area Agency on Aging", "AAA", "AARP",
    "senior center", "nutrition site",

    # Disabilities
    "disability", "disabilities", "disabled",
    "ADA", "accessibility", "accessible", "accommodation",
    "special needs", "special education",
    "developmental disability", "DD", "intellectual disability", "ID",
    "autism", "autistic", "ASD", "spectrum",
    "cerebral palsy", "CP", "down syndrome",
    "blind", "visually impaired", "low vision",
    "hard of hearing", "hearing impaired",
    "mobility", "mobility impaired", "paralysis", "spinal cord",
    "regional center", "DDS",
    "SSI", "SSDI", "social security disability",
    "independent living", "ILC",
    "respite", "respite care",

    # Legal
    "legal", "lawyer", "attorney", "law", "court",
    "legal aid", "legal services", "pro bono", "free legal",
    "civil", "criminal", "family law", "immigration law",
    "expungement", "record clearance", "clean slate",
    "tenant rights", "eviction defense", "unlawful detainer",
    "divorce", "child support", "restraining order",
    "public defender", "self help", "self-help",

    # Immigration
    "immigration", "immigrant", "migrant",
    "refugee", "asylee", "asylum",
    "citizenship", "naturalization", "civics",
    "visa", "green card", "DACA", "dreamer",
    "TPS", "temporary protected status",
    "deportation", "removal", "ICE",
    "undocumented", "documentation",
    "resettlement", "IRC", "refugee resettlement",

    # Domestic Violence & Abuse
    "domestic violence", "DV", "intimate partner violence", "IPV",
    "abuse", "abused", "abuser", "batterer",
    "victim", "survivor", "battered",
    "protective order",
    "safe house", "safe haven",
    "hotline", "crisis line",
    "sexual assault", "rape", "SART",
    "child abuse", "elder abuse", "neglect",
    "human trafficking", "trafficking", "exploitation",

    # Veterans
    "veteran", "veterans", "vet", "vets",
    "military", "armed forces", "service member",
    "army", "navy", "air force", "marines", "coast guard",
    "VA", "veterans affairs", "veterans administration",
    "combat", "deployment",
    "GI bill", "veterans benefits",
    "VFW", "American Legion", "DAV",

    # Crisis & Emergency
    "crisis", "emergency", "urgent",
    "helpline", "warm line",
    "suicide", "suicidal", "988", "suicide prevention",
    "intervention", "stabilization",
    "disaster", "fire", "flood", "earthquake", "wildfire",
    "FEMA", "Red Cross", "disaster relief",

    # Financial
    "financial", "finance", "money", "cash",
    "budget", "budgeting", "financial literacy",
    "debt", "credit", "credit repair", "bankruptcy",
    "tax", "taxes", "VITA", "tax preparation", "free tax",
    "EITC", "earned income tax credit", "child tax credit",
    "benefits", "public benefits", "government benefits",
    "CalFresh", "SNAP", "food stamps", "EBT",
    "CalWORKs", "TANF", "welfare", "cash aid",
    "General Relief", "GR", "GA", "general assistance",
    "Medi-Cal", "Medicaid", "health insurance",
    "Covered California", "ACA", "Obamacare", "marketplace",
    "social security",
    "assistance", "aid", "help",

    # Transportation
    "transportation", "transport", "ride", "rides",
    "bus", "trolley", "MTS", "transit", "public transit",
    "paratransit", "dial a ride", "ACCESS",
    "car", "vehicle", "auto", "automobile",
    "gas money", "fuel",
    "driver license", "DMV",

    # Technology
    "internet", "wifi", "broadband", "connectivity",
    "computer", "laptop", "tablet", "device",
    "digital", "digital literacy", "computer class",
    "phone", "cell phone", "smartphone", "lifeline",

    # Language & Cultural
    "Spanish", "Espanol", "bilingual", "bicultural",
    "translation", "interpretation", "interpreter",
    "English learner",
    "Latino", "Hispanic", "Chicano",
    "Asian", "Pacific Islander", "API", "AAPI",
    "African American", "Black",
    "Native American", "Indigenous", "tribal",
    "newcomer",
    "multicultural", "cultural", "ethnic",

    # LGBTQ+
    "LGBTQ", "LGBT", "LGBTQIA",
    "gay", "lesbian", "bisexual", "transgender", "trans",
    "queer", "questioning", "nonbinary",
    "pride", "coming out",

    # Faith-based
    "church", "faith", "faith-based", "religious",
    "Christian", "Catholic", "Protestant", "Baptist", "Methodist",
    "Jewish", "synagogue", "Muslim", "mosque", "Buddhist", "temple",
    "Salvation Army", "Catholic Charities", "Lutheran",
    "St. Vincent de Paul", "Goodwill",

    # Organizations & Programs
    "211", "2-1-1", "United Way",
    "Boys Girls Club",
    "Big Brothers Big Sisters",
    "Habitat for Humanity", "Habitat",
    "community action", "CAP",
    "CalAIM", "community supports",

    # Service types
    "program", "programs", "service", "services",
    "center", "resource center", "community center",
    "agency", "organization", "nonprofit", "non-profit",
    "charity", "charitable",
    "outreach", "case management", "navigation",
    "referral", "intake", "assessment", "enrollment",
    "application", "apply", "sign up", "register",
    "free", "no cost", "sliding scale",
    "walk-in", "appointment", "same day",

    # Location-based
    "San Diego", "SD", "county", "city",
    "downtown", "El Cajon", "Escondido", "Oceanside",
    "Chula Vista", "National City", "Vista", "Carlsbad",
    "La Mesa", "Santee", "Poway", "Encinitas",
    "north county", "south bay", "east county",

    # Common terms
    "support", "resource", "resources",
    "information", "connect", "access",
    "need", "needs", "basic needs",
    "low income", "low-income", "poverty", "poor",
    "unhoused", "houseless",
    "at risk", "at-risk", "vulnerable",

    # Single letters (to catch unique matches)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",

    # Spanish terms
    "comida", "alimentos", "ropa", "vivienda", "alquiler",
    "salud", "medico", "clinica", "doctor",
    "trabajo", "empleo", "educacion", "escuela",
    "familia", "ninos", "ancianos", "discapacidad",
    "inmigracion", "ciudadania",
    "ayuda", "asistencia", "servicios", "recursos",
    "gratis", "gratuito", "bajo costo",
    "emergencia",

    # Catch-all
    "*",
]


def get_unique_keywords():
    """Remove duplicate keywords while preserving order."""
    seen = set()
    unique = []
    for kw in SEARCH_KEYWORDS:
        kw_lower = kw.lower().strip()
        if kw_lower and kw_lower not in seen:
            seen.add(kw_lower)
            unique.append(kw)
    return unique


def fetch_all_services(client: APIClient, location: str = "san diego", rate_limit: float = 0.2):
    """
    Fetch all services using keyword expansion.

    Args:
        client: API client instance
        location: Location to search (default: san diego)
        rate_limit: Seconds to wait between requests

    Returns:
        Tuple of (results list, stats dict)
    """
    search_headers = {
        "searchWithinLocationType": "County",
        "searchMode": "All",
        "locationMode": "Within",
        "keywordIsTaxonomyCode": "false",
        "keywordIsTaxonomyTerm": "false",
        "resultsAdvanced": "false",
        "orderByDistance": "true",
    }

    # Get total count
    params = {"keywords": "*", "location": location, "size": 100, "skip": 0}
    response = client.get(SEARCH_ENDPOINT, params=params, extra_headers=search_headers)
    data = response.json()
    total_available = data.get("count", 0)

    print(f"Total services reported by API: {total_available}")
    print(f"Starting fetch at: {time.strftime('%H:%M:%S')}")

    keywords = get_unique_keywords()
    print(f"Searching with {len(keywords)} unique keywords")

    all_results = []
    seen_ids = set()
    keyword_stats = []

    fetch_start = time.time()

    for i, keyword in enumerate(tqdm(keywords, desc="Fetching")):
        try:
            params = {"keywords": keyword, "location": location, "size": 100, "skip": 0}
            response = client.get(SEARCH_ENDPOINT, params=params, extra_headers=search_headers)
            data = response.json()

            results = data.get("results", [])
            count = data.get("count", 0)

            new_count = 0
            for r in results:
                sal_id = r.get("idServiceAtLocation")
                if sal_id and sal_id not in seen_ids:
                    seen_ids.add(sal_id)
                    all_results.append(r)
                    new_count += 1

            keyword_stats.append({
                "keyword": keyword,
                "api_count": count,
                "fetched": len(results),
                "new_unique": new_count,
            })

            # Always log progress
            print(f"[{i+1}/{len(keywords)}] '{keyword}': {count} results, +{new_count} new | Total: {len(all_results)}/{total_available} ({100*len(all_results)/total_available:.1f}%)", flush=True)

            if len(seen_ids) >= total_available:
                tqdm.write(f"\n*** REACHED TARGET: {len(seen_ids)}/{total_available} services ***")
                break

            time.sleep(rate_limit)

        except Exception as e:
            tqdm.write(f"  Error with '{keyword}': {e}")
            time.sleep(2)

    fetch_duration = time.time() - fetch_start

    stats = {
        "total_available": total_available,
        "total_fetched": len(all_results),
        "unique_ids": len(seen_ids),
        "keywords_searched": i + 1,
        "fetch_duration_seconds": fetch_duration,
        "coverage_percent": 100 * len(seen_ids) / total_available if total_available > 0 else 0,
    }

    return all_results, stats, keyword_stats


def fetch_service_details(client: APIClient, service_ids: list, cache: ResponseCache):
    """Fetch detailed information for each service."""
    details = {}
    cache_hits = 0
    api_calls = 0

    for sal_id in tqdm(service_ids, desc="Fetching details"):
        cache_key = f"detail_{sal_id}"

        cached = cache.get(cache_key)
        if cached is not None:
            details[sal_id] = cached
            cache_hits += 1
            continue

        try:
            url = f"{DETAIL_ENDPOINT}/{sal_id}"
            response = client.get(url)
            data = response.json()

            cache.set(cache_key, data)
            details[sal_id] = data
            api_calls += 1

            time.sleep(0.1)

        except requests.exceptions.HTTPError as e:
            print(f"Error fetching {sal_id}: {e}")
            details[sal_id] = {"error": str(e), "id": sal_id}

    print(f"Fetched {len(details)} details (cache hits: {cache_hits}, API calls: {api_calls})")
    return details


def flatten_record(result: dict, detail: Optional[dict] = None) -> dict:
    """Flatten a search result into a single row."""
    row = {}

    row["service_at_location_id"] = result.get("idServiceAtLocation")
    row["organization_id"] = result.get("idOrganization")
    row["service_id"] = result.get("idService")
    row["location_id"] = result.get("idLocation")

    row["organization_name"] = result.get("nameOrganization")
    row["service_name"] = result.get("nameService")
    row["location_name"] = result.get("nameLocation")

    row["organization_description"] = result.get("descriptionOrganization")
    row["service_description"] = result.get("descriptionService")

    addr = result.get("address") or {}
    row["address_street"] = addr.get("streetAddress")
    row["city"] = addr.get("city")
    row["county"] = addr.get("county")
    row["state"] = addr.get("stateProvince")
    row["postal_code"] = addr.get("postalCode")
    row["latitude"] = addr.get("latitude")
    row["longitude"] = addr.get("longitude")

    taxonomies = result.get("taxonomy") or []
    if taxonomies:
        row["taxonomy_codes"] = ", ".join([t.get("taxonomyCode", "") for t in taxonomies if t.get("taxonomyCode")])
        row["taxonomy_terms"] = ", ".join([t.get("taxonomyTerm", "") for t in taxonomies if t.get("taxonomyTerm")])

    row["data_owner"] = result.get("dataOwner")
    row["status"] = result.get("status")

    if detail and "error" not in detail:
        if detail.get("descriptionService"):
            row["service_description"] = detail.get("descriptionService")
        row["application_process"] = detail.get("applicationProcess")

    return row


def build_dataframe(results: list, details: dict) -> pd.DataFrame:
    """Build DataFrame from results and details."""
    rows = []
    for result in results:
        sal_id = str(result.get("idServiceAtLocation") or "")
        detail = details.get(sal_id)
        row = flatten_record(result, detail)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["service_at_location_id"], keep="first")
    return df


def save_performance_metrics(stats: dict, keyword_stats: list, df: pd.DataFrame):
    """Save performance metrics to CSV."""
    top_keywords = sorted(keyword_stats, key=lambda x: x["new_unique"], reverse=True)[:15]

    lines = [
        "Performance Metrics",
        "Metric,Value",
        f"Coverage,{stats['unique_ids']}/{stats['total_available']} ({stats['coverage_percent']:.1f}%)",
        f"Keywords searched,{stats['keywords_searched']}",
        f"API requests made,{stats['keywords_searched']}",
        f"Fetch duration,{stats['fetch_duration_seconds']:.2f} seconds ({stats['fetch_duration_seconds']/60:.2f} minutes)",
        "",
        "Top Contributing Keywords",
        "Keyword,New Services",
    ]

    for k in top_keywords:
        if k["new_unique"] > 0:
            lines.append(f"{k['keyword']},{k['new_unique']}")

    lines.extend([
        "",
        "Services by City",
        "City,Count",
    ])

    city_counts = df["city"].value_counts()
    for city, count in city_counts.items():
        lines.append(f"{city},{count}")

    with open(OUTPUT_DIR / "performance_metrics.csv", "w") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 60)
    print("211 API SCRAPER - San Diego County")
    print("=" * 60)

    start_time = time.time()

    client = APIClient(API_KEY)
    cache = ResponseCache(CACHE_DIR)

    # Fetch all services
    results, stats, keyword_stats = fetch_all_services(client, location="san diego")

    print(f"\n{'=' * 60}")
    print("FETCH COMPLETE")
    print(f"{'=' * 60}")
    print(f"Keywords searched: {stats['keywords_searched']}")
    print(f"Total services fetched: {stats['total_fetched']}")
    print(f"Coverage: {stats['unique_ids']}/{stats['total_available']} ({stats['coverage_percent']:.1f}%)")
    print(f"Fetch duration: {stats['fetch_duration_seconds']:.2f} seconds")

    # Save raw results
    with open(OUTPUT_DIR / "raw_search_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Extract service IDs
    service_ids = [r.get("idServiceAtLocation") for r in results if r.get("idServiceAtLocation")]

    # Fetch details
    print("\nFetching service details...")
    details = fetch_service_details(client, service_ids, cache)

    # Build DataFrame
    print("\nBuilding DataFrame...")
    df = build_dataframe(results, details)

    # Save outputs
    df.to_csv(OUTPUT_DIR / "services.csv", index=False)
    print(f"Saved {len(df)} services to output/services.csv")

    # Save ID list
    id_df = df[["service_at_location_id", "service_name", "organization_name", "city"]].copy()
    id_df.to_csv(OUTPUT_DIR / "idServiceAtLocation_list.csv", index=False)
    print(f"Saved ID list to output/idServiceAtLocation_list.csv")

    # Save metrics
    save_performance_metrics(stats, keyword_stats, df)
    print(f"Saved metrics to output/performance_metrics.csv")

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()
