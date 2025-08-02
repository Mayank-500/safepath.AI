import pandas as pd
import folium
import os
import networkx as nx
from scipy.spatial import distance
import geopandas as gpd
from geopy.distance import geodesic  # For accurate distance calculation

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data[data['route_id'] <= 100]  # Filter dataset to first 50 routes (adjust as needed)
    
    # Normalizing features
    features = ['crime_density', 'lighting_density', 'surveillance_score', 'police_proximity', 
                'emergency_services', 'population_density', 'user_feedback', 'public_transport_nearby', 
                'weather_conditions']
    for feature in features:
        data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

    # Calculating safety score
    weights = {
        'crime_density': 0.2, 'lighting_density': 0.1, 'surveillance_score': 0.1,
        'police_proximity': 0.1, 'emergency_services': 0.1, 'population_density': 0.1,
        'user_feedback': 0.1, 'public_transport_nearby': 0.1, 'weather_conditions': 0.1
    }

    data['safety_score'] = sum(data[feature] * weight for feature, weight in weights.items())
    return data

# Calculate safest and fastest route
def safest_and_fastest_route(data, start_id, end_id, alpha=0.7, beta=0.3):
    G = nx.Graph()
    
    for _, row in data.iterrows():
        G.add_node(row['route_id'], safety_score=row['safety_score'], latitude=row['latitude'], longitude=row['longitude'])
    
    for i in range(len(data) - 1):
        route1, route2 = data.iloc[i], data.iloc[i + 1]
        dist = distance.euclidean((route1['latitude'], route1['longitude']),
                                  (route2['latitude'], route2['longitude']))
        safety_factor = 1 / min(route1['safety_score'], route2['safety_score'])
        weight = alpha * dist + beta * safety_factor
        G.add_edge(route1['route_id'], route2['route_id'], weight=weight, distance=dist)
    
    if start_id not in G.nodes or end_id not in G.nodes:
        raise ValueError(f"Node {start_id} or {end_id} not found in graph.")
    
    return nx.dijkstra_path(G, start_id, end_id, weight='weight')

# Visualize the safest and unsafe routes on the map
def visualize_safest_and_unsafe_routes_on_map(data, safest_route, output_path, average_car_speed=30):
    m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=14)

    # Get the minimum safety score from the safest path
    min_safe_score = min(data.loc[data['route_id'].isin(safest_route), 'safety_score'])

    # Add markers with full parameter details
    for _, row in data.iterrows():
        is_safest = row['route_id'] in safest_route
        color = 'blue' if is_safest else ('red' if row['safety_score'] < min_safe_score else None)

        if color:  # Ensures only blue or red markers are added (no black/gray)
            popup_text = "<strong>Route Details:</strong><br>"
            popup_text += f"Route ID: {row['route_id']}<br>Safety Score: {row['safety_score']:.2f}<br>"
            popup_text += f"Crime Density: {row['crime_density']:.2f}<br>"
            popup_text += f"Lighting Density: {row['lighting_density']:.2f}<br>"
            popup_text += f"Surveillance Score: {row['surveillance_score']:.2f}<br>"
            popup_text += f"Police Proximity: {row['police_proximity']:.2f}<br>"
            popup_text += f"Emergency Services: {row['emergency_services']:.2f}<br>"
            popup_text += f"Population Density: {row['population_density']:.2f}<br>"
            popup_text += f"User Feedback: {row['user_feedback']:.2f}<br>"
            popup_text += f"Public Transport Nearby: {row['public_transport_nearby']:.2f}<br>"
            popup_text += f"Weather Conditions: {row['weather_conditions']:.2f}<br>"

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color=color),
                tooltip=f"Route ID: {row['route_id']}"
            ).add_to(m)

    # Create a zigzag path for the safest route
    route_coords = [
        (data.loc[data['route_id'] == route_id, 'latitude'].values[0],
         data.loc[data['route_id'] == route_id, 'longitude'].values[0])
        for route_id in safest_route
    ]

    if len(route_coords) > 1:
        zigzag_coords = []
        for i in range(len(route_coords) - 1):
            lat1, lon1 = route_coords[i]
            lat2, lon2 = route_coords[i + 1]
            mid_lat, mid_lon = (lat1 + lat2) / 2, (lon1 + lon2) / 2 + 0.0005  # Zigzag effect
            zigzag_coords.extend([(lat1, lon1), (mid_lat, mid_lon), (lat2, lon2)])

        folium.PolyLine(zigzag_coords, color="blue", weight=6, opacity=0.8, tooltip="Safest Route").add_to(m)

    # Calculate and display route info
    total_distance = sum(geodesic(route_coords[i], route_coords[i+1]).km for i in range(len(route_coords) - 1))
    travel_time_minutes = (total_distance / average_car_speed) * 60 if average_car_speed > 0 else 0

    folium.Marker(
        location=route_coords[len(route_coords)//2],
        popup=f"Total Distance: {total_distance:.2f} km<br>Estimated Travel Time: {travel_time_minutes:.0f} minutes",
        icon=folium.Icon(color='green', icon='info-sign'),
        tooltip="Route Information"
    ).add_to(m)

    m.save(output_path)
    print(f"Map with safest route saved at: {output_path}")

# Convert data to GeoJSON
def convert_to_geojson(data, output_path):
    gdf = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data['longitude'], data['latitude']))
    gdf.to_file(output_path, driver="GeoJSON")
    print(f"GeoJSON file saved at: {output_path}")

# File path for dataset
file_path = "/Users/mayankchaudhary/Documents/safepath/safest_route_dataset_delhi-2.csv"

# Load and preprocess data
data = load_and_preprocess_data(file_path)

# Get start and end route IDs from the user
start_route_id = int(input("Enter the starting route ID: ").strip())
end_route_id = int(input("Enter the ending route ID: ").strip())

# Get the safest route
safest_path = safest_and_fastest_route(data, start_route_id, end_route_id, alpha=0.7, beta=0.3)

# Output directory
output_directory = "./output"
os.makedirs(output_directory, exist_ok=True)

# Save GeoJSON output
geojson_output_path = os.path.join(output_directory, "safest_routes.geojson")
convert_to_geojson(data, geojson_output_path)

# Save HTML map with route info
html_output_path = os.path.join(output_directory, "safest_route_map.html")
visualize_safest_and_unsafe_routes_on_map(data, safest_path, html_output_path, average_car_speed=30)
