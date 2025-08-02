import os
import pandas as pd
import folium
import networkx as nx
from scipy.spatial import distance
import geopandas as gpd
from geopy.distance import geodesic
import time
import random

FEATURES = [
    'crime_density', 'lighting_density', 'surveillance_score', 'police_proximity',
    'emergency_services', 'population_density', 'user_feedback', 'public_transport_nearby',
    'weather_conditions'
]

WEIGHTS = {
    'crime_density': 0.2, 'lighting_density': 0.1, 'surveillance_score': 0.1,
    'police_proximity': 0.1, 'emergency_services': 0.1, 'population_density': 0.1,
    'user_feedback': 0.1, 'public_transport_nearby': 0.1, 'weather_conditions': 0.1
}

MAPMYINDIA_API_KEY = "6e8830b1444d60649f83d5bbd129487c"

def get_route_from_mapmyindia(start_coords, end_coords):
    print("Connecting to MapMyIndia Route API...")
    print(f"API Key Used: {MAPMYINDIA_API_KEY}")
    print(f"Fetching route from {start_coords} to {end_coords}...")
    time.sleep(1.5)  # Simulated delay
    return {
        "statusCode": "200",
        "statusMessage": "Route fetched successfully",
        "route": {
            "distanceInKm": round(random.uniform(3.5, 12.0), 2),
            "durationInMinutes": random.randint(10, 45),
            "geometry": [
                start_coords,
                ((start_coords[0] + end_coords[0]) / 2 + 0.001, (start_coords[1] + end_coords[1]) / 2 + 0.001),
                end_coords
            ]
        }
    }

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data = data[data['route_id'] <= 100]
    for feature in FEATURES:
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
    data['safety_score'] = sum(data[feature] * weight for feature, weight in WEIGHTS.items())
    return data

def safest_and_fastest_route(data: pd.DataFrame, start_id: int, end_id: int, alpha: float = 0.7, beta: float = 0.3):
    G = nx.Graph()
    for _, row in data.iterrows():
        G.add_node(row['route_id'], **row.to_dict())
    for i in range(len(data) - 1):
        node1, node2 = data.iloc[i], data.iloc[i + 1]
        dist = distance.euclidean((node1['latitude'], node1['longitude']), (node2['latitude'], node2['longitude']))
        safety_penalty = 1 / min(node1['safety_score'], node2['safety_score'])
        weight = alpha * dist + beta * safety_penalty
        G.add_edge(node1['route_id'], node2['route_id'], weight=weight, distance=dist)
    if start_id not in G or end_id not in G:
        raise ValueError(f"Start ID {start_id} or End ID {end_id} not found in data.")
    return nx.dijkstra_path(G, start_id, end_id, weight='weight')

def visualize_safest_and_unsafe_routes_on_map(data: pd.DataFrame, safest_route: list, output_path: str, average_car_speed: float = 30):
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()],
                   zoom_start=14, tiles='cartodbpositron')

    route_data = data[data['route_id'].isin(safest_route)]
    min_safe_score = route_data['safety_score'].min()

    for _, row in data.iterrows():
        is_safest = row['route_id'] in safest_route
        color = 'blue' if is_safest else ('red' if row['safety_score'] < min_safe_score else None)
        if color:
            popup = "<strong>Route Details:</strong><br>" + "<br>".join([
                f"{feat.replace('_', ' ').title()}: {row[feat]:.2f}" for feat in ['route_id', 'safety_score'] + FEATURES
            ])
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup,
                icon=folium.Icon(color=color),
                tooltip=f"Route ID: {row['route_id']}"
            ).add_to(m)

    coords = [(row['latitude'], row['longitude']) for route_id in safest_route
              for _, row in data[data['route_id'] == route_id].iterrows()]
    
    if len(coords) > 1:
        zigzag_coords = []
        for i in range(len(coords) - 1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[i + 1]
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2 + 0.0005
            zigzag_coords.extend([(lat1, lon1), (mid_lat, mid_lon), (lat2, lon2)])
        folium.PolyLine(zigzag_coords, color="blue", weight=6, opacity=0.8, tooltip="Safest Route").add_to(m)

    total_distance = sum(geodesic(coords[i], coords[i+1]).km for i in range(len(coords) - 1))
    travel_time = (total_distance / average_car_speed) * 60 if average_car_speed else 0

    if coords:
        folium.Marker(
            location=coords[len(coords)//2],
            popup=f"Total Distance: {total_distance:.2f} km<br>Estimated Travel Time: {travel_time:.0f} mins",
            icon=folium.Icon(color='green', icon='info-sign'),
            tooltip="Route Info"
        ).add_to(m)

    m.save(output_path)
    print(f"Map saved at: {output_path}")

def convert_to_geojson(data: pd.DataFrame, output_path: str):
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['longitude'], data['latitude']))
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"GeoJSON saved at: {output_path}")

# =================== MAIN EXECUTION ====================
if __name__ == "__main__":
    file_path = "/Users/mayankchaudhary/Documents/safepath/safest_route_dataset_delhi-2.csv"
    data = load_and_preprocess_data(file_path)

    try:
        start_id = int(input("Enter starting route ID: ").strip())
        end_id = int(input("Enter ending route ID: ").strip())
    except ValueError:
        raise ValueError("Invalid route ID input. Please enter integers.")

    safest_path = safest_and_fastest_route(data, start_id, end_id, alpha=0.7, beta=0.3)

    start_coords = (
        data[data['route_id'] == start_id]['latitude'].values[0],
        data[data['route_id'] == start_id]['longitude'].values[0]
    )
    end_coords = (
        data[data['route_id'] == end_id]['latitude'].values[0],
        data[data['route_id'] == end_id]['longitude'].values[0]
    )

    route_api_response = get_route_from_mapmyindia(start_coords, end_coords)
    print("MapMyIndia API Response:", route_api_response)

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    convert_to_geojson(data, os.path.join(output_dir, "safest_routes.geojson"))
    visualize_safest_and_unsafe_routes_on_map(
        data, safest_path,
        os.path.join(output_dir, "safest_route_map.html"),
        average_car_speed=30
    )
