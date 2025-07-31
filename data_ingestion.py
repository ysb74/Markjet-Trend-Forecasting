import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import time
import random
from datetime import datetime, timedelta

# --- Configuration for Real-World APIs/Systems ---
ZILLOW_API_KEY = "YOUR_ZILLOW_API_KEY"
REALTOR_API_KEY = "YOUR_REALTOR_API_KEY"
ZILLOW_PROPERTY_API_ENDPOINT = "https://api.zillow.com/v1/property_details"
REALTOR_LISTINGS_API_ENDPOINT = "https://api.realtor.com/v1/listings"
CRM_API_ENDPOINT = "https://your-crm-instance.com/api/v2/tenants"
CRM_AUTH_TOKEN = "YOUR_CRM_AUTH_TOKEN"

def load_crm_data_live():
    """
    Simulates loading tenant and property data directly from a CRM system's API.
    Returns:
        pd.DataFrame: A DataFrame containing CRM data (simulated for this example).
    """
    print(f"--- Attempting to load CRM data from {CRM_API_ENDPOINT} ---")
    try:
        crm_raw_data = {
            "tenants": [
                {'tenant_id': 101, 'lease_start': '2023-01-15', 'lease_end': '2024-01-15', 'rent': 1500, 'payments_late_6m': 0, 'maint_req_12m': 1, 'feedback_score': 4.5, 'prop_id': 'P001', 'status': 'Active', 'lease_renewal_status': 'Renewed'},
                {'tenant_id': 102, 'lease_start': '2022-07-01', 'lease_end': '2024-07-01', 'rent': 2000, 'payments_late_6m': 1, 'maint_req_12m': 2, 'feedback_score': 3.8, 'prop_id': 'P002', 'status': 'Active', 'lease_renewal_status': 'Renewed'},
                {'tenant_id': 103, 'lease_start': '2023-03-20', 'lease_end': '2024-03-20', 'rent': 1200, 'payments_late_6m': 3, 'maint_req_12m': 4, 'feedback_score': 2.5, 'prop_id': 'P001', 'status': 'Active', 'lease_renewal_status': 'Churned'},
                {'tenant_id': 104, 'lease_start': '2024-01-10', 'lease_end': '2025-01-10', 'rent': 1800, 'payments_late_6m': 0, 'maint_req_12m': 0, 'feedback_score': 4.9, 'prop_id': 'P003', 'status': 'Active', 'lease_renewal_status': 'Renewed'},
                {'tenant_id': 105, 'lease_start': '2022-11-05', 'lease_end': '2024-11-05', 'rent': 2500, 'payments_late_6m': 2, 'maint_req_12m': 3, 'feedback_score': 3.1, 'prop_id': 'P002', 'status': 'Active', 'lease_renewal_status': 'Churned'},
            ]
        }
        df_crm = pd.DataFrame(crm_raw_data.get("tenants", []))
        df_crm.rename(columns={
            'lease_start': 'lease_start_date',
            'lease_end': 'lease_end_date',
            'rent': 'monthly_rent',
            'payments_late_6m': 'payment_delays_last_6_months',
            'maint_req_12m': 'maintenance_requests_last_year',
            'prop_id': 'property_id',
            'lease_renewal_status': 'lease_renewal'
        }, inplace=True)
        df_crm['lease_renewal'] = df_crm['lease_renewal'].map({'Churned': 0, 'Renewed': 1})
        print("Successfully loaded CRM data via simulated API.")
        return df_crm
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to CRM API: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred processing CRM data: {e}")
        return pd.DataFrame()

def scrape_public_listings(search_url='http://quotes.toscrape.com/', num_pages=1):
    """
    Conceptual function to scrape publicly available property listing data.
    """
    scraped_data = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    print(f"--- Attempting to scrape public listings from {search_url} ---")
    for page in range(1, num_pages + 1):
        page_url = f"{search_url}/page/{page}/" if num_pages > 1 else search_url
        try:
            response = requests.get(page_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            simulated_listing_cards = [
                {'address': f'123 Main St, Anytown - Scraped {page}', 'price_str': '$355,000', 'beds_str': '3 beds', 'baths_str': '2 baths', 'sqft_str': '1,500 sqft'},
                {'address': f'456 Oak Ave, Anytown - Scraped {page}', 'price_str': '$510,000', 'beds_str': '4 beds', 'baths_str': '3 baths', 'sqft_str': '2,250 sqft'},
            ]
            for listing in simulated_listing_cards:
                scraped_data.append({
                    'scraped_address': listing['address'],
                    'scraped_price_raw': listing['price_str'],
                    'scraped_beds_raw': listing['beds_str'],
                    'scraped_baths_raw': listing['baths_str'],
                    'scraped_sqft_raw': listing['sqft_str'],
                    'scraped_source_url': page_url
                })
            print(f"Scraped page {page} successfully (conceptual).")
            time.sleep(random.uniform(1, 3))
        except requests.exceptions.RequestException as e:
            print(f"Error scraping page {page_url}: {e}")
            break
        except Exception as e:
            print(f"An unexpected error occurred during scraping: {e}")
            break
    return scraped_data

def get_property_data_from_official_api(api_name='Zillow', location='Anytown', property_type='house'):
    """
    Simulates fetching property data from official Zillow or Realtor.com APIs.
    """
    api_key = ZILLOW_API_KEY if api_name == 'Zillow' else REALTOR_API_KEY
    api_endpoint = ZILLOW_PROPERTY_API_ENDPOINT if api_name == 'Zillow' else REALTOR_LISTINGS_API_ENDPOINT
    print(f"--- Attempting to fetch data from {api_name} API for {location} ---")
    if api_key == "YOUR_ZILLOW_API_KEY" or api_key == "YOUR_REALTOR_API_KEY":
        print(f"Warning: Using placeholder API key for {api_name}. Real API calls will fail without a genuine key.")
    try:
        if api_name == 'Zillow':
            api_raw_data = {
                "properties": [
                    {"zpid": "P001", "address": {"street": "123 Main St", "city": "Anytown"}, "bedrooms": 3, "bathrooms": 2, "squareFootage": 1500, "zestimate": 350500, "lastSoldDate": "2020-05-10"},
                    {"zpid": "P002", "address": {"street": "456 Oak Ave", "city": "Anytown"}, "bedrooms": 4, "bathrooms": 3, "squareFootage": 2200, "zestimate": 505000, "lastSoldDate": "2021-01-20"},
                    {"zpid": "P003", "address": {"street": "789 Pine Ln", "city": "Anytown"}, "bedrooms": 2, "bathrooms": 1, "squareFootage": 900, "zestimate": 200100, "lastSoldDate": "2019-11-15"}
                ],
                "status": "success"
            }
        elif api_name == 'Realtor':
            api_raw_data = {
                "listings": [
                    {"listing_id": "L001", "location": {"address": "123 Main St, Anytown"}, "beds": 3, "baths": 2, "sq_ft": 1500, "list_price": 350000, "sale_date": "2020-05-10"},
                    {"listing_id": "L002", "location": {"address": "456 Oak Ave, Anytown"}, "beds": 4, "baths": 3, "sq_ft": 2200, "list_price": 500000, "sale_date": "2021-01-20"},
                    {"listing_id": "L003", "location": {"address": "789 Pine Ln, Anytown"}, "beds": 2, "baths": 1, "sq_ft": 900, "list_price": 200000, "sale_date": "2019-11-15"}
                ],
                "status": "success"
            }
        else:
            api_raw_data = {"status": "error", "message": "Invalid API name"}

        if api_raw_data.get("status") == "success":
            if api_name == 'Zillow':
                normalized_data = []
                for prop in api_raw_data.get("properties", []):
                    normalized_data.append({
                        'api_source': 'Zillow',
                        'property_api_id': prop.get('zpid'),
                        'address': f"{prop.get('address', {}).get('street')}, {prop.get('address', {}).get('city')}",
                        'beds': prop.get('bedrooms'),
                        'baths': prop.get('bathrooms'),
                        'sqft': prop.get('squareFootage'),
                        'price': prop.get('zestimate'),
                        'last_sold_date': prop.get('lastSoldDate')
                    })
                print(f"Successfully fetched and normalized data from {api_name} API.")
                return normalized_data
            elif api_name == 'Realtor':
                normalized_data = []
                for listing in api_raw_data.get("listings", []):
                    normalized_data.append({
                        'api_source': 'Realtor',
                        'property_api_id': listing.get('listing_id'),
                        'address': listing.get('location', {}).get('address'),
                        'beds': listing.get('beds'),
                        'baths': listing.get('baths'),
                        'sqft': listing.get('sq_ft'),
                        'price': listing.get('list_price'),
                        'last_sold_date': listing.get('sale_date')
                    })
                print(f"Successfully fetched and normalized data from {api_name} API.")
                return normalized_data
        else:
            print(f"{api_name} API call failed: {api_raw_data.get('message', 'Unknown error')}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Error during {api_name} API call: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred processing {api_name} API data: {e}")
        return []

def aggregate_real_estate_data(crm_df, scraped_data_list, api_data_list):
    """
    Aggregates data from different sources into a unified DataFrame.
    """
    print("\n--- Aggregating Data from Multiple Sources ---")
    df_scraped = pd.DataFrame(scraped_data_list)
    df_api = pd.DataFrame(api_data_list)
    unified_df = crm_df.copy()
    print(f"Starting with CRM data. Shape: {unified_df.shape}")

    if not df_api.empty:
        df_api['property_id'] = df_api['property_api_id']
        api_cols_to_merge = ['property_id', 'beds', 'baths', 'sqft', 'price', 'last_sold_date']
        df_api_filtered = df_api[api_cols_to_merge].drop_duplicates(subset=['property_id'])
        unified_df = pd.merge(unified_df, df_api_filtered, on='property_id', how='left', suffixes=('_crm', '_api'))
        print(f"Merged with API data. New shape: {unified_df.shape}")
    else:
        print("No API data to merge.")

    if not df_scraped.empty:
        print(f"Scraped data available. Shape: {df_scraped.shape}. Manual or fuzzy matching may be needed.")
        unified_df['has_scraped_listing'] = unified_df['property_id'].apply(
            lambda pid: 1 if any(f"P{i}" in s['scraped_address'] for i, s in enumerate(scraped_data_list)) else 0
        )
        print("Added 'has_scraped_listing' indicator (conceptual).")
    else:
        print("No scraped data to integrate.")

    for col in ['beds', 'baths', 'sqft', 'price']:
        if f'{col}_api' in unified_df.columns and f'{col}_crm' in unified_df.columns:
            unified_df[col] = unified_df[f'{col}_api'].fillna(unified_df[f'{col}_crm'])
            unified_df.drop(columns=[f'{col}_crm', f'{col}_api'], inplace=True)
        elif f'{col}_api' in unified_df.columns:
            unified_df.rename(columns={f'{col}_api': col}, inplace=True)
        elif f'{col}_crm' in unified_df.columns:
            unified_df.rename(columns={f'{col}_crm': col}, inplace=True)
        if col in unified_df.columns:
            unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce')
            unified_df[col].fillna(unified_df[col].median(), inplace=True)

    if 'last_sold_date' in unified_df.columns:
        unified_df['last_sold_date'] = pd.to_datetime(unified_df['last_sold_date'], errors='coerce')
        unified_df['days_since_last_sold'] = (pd.to_datetime('now') - unified_df['last_sold_date']).dt.days.fillna(0)
    
    unified_df.drop(columns=['api_source', 'property_api_id'] if 'api_source' in unified_df.columns else [], inplace=True, errors='ignore')

    print(f"Final unified DataFrame shape: {unified_df.shape}")
    return unified_df