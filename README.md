# 211 API Scraper

Fetches all services from the 211 API for San Diego County using keyword expansion to work around the API's pagination limitation (max 10 results per query).

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API_211_KEY
   ```

3. **Run the scraper:**
   ```bash
   python scraper.py
   ```

## Output Files

| File | Description |
|------|-------------|
| `output/services.csv` | Full service data with all fields |
| `output/idServiceAtLocation_list.csv` | Just IDs with basic info |
| `output/performance_metrics.csv` | Scraping performance stats |
| `output/raw_search_results.json` | Raw API responses |

## Configuration

To change the location, edit `scraper.py` and modify the `location` parameter in the `main()` function:

```python
results, stats, keyword_stats = fetch_all_services(client, location="your city")
```

**Important:** The `searchWithinLocationType` header must be set to `"County"` for full results.

## Performance

- **San Diego County:** 277 services, ~30 minutes
- Uses 789 unique keywords to maximize coverage
- Rate limited to 0.2 seconds between requests
- Caches detail API responses to avoid duplicate calls

## API Notes

- The 211 API returns max 10 results per query
- The `skip` parameter for pagination doesn't work
- Solution: Use keyword expansion to get different result sets
- Each unique `idServiceAtLocation` is deduplicated
