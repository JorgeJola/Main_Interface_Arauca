from flask import Blueprint, Flask,render_template, jsonify, send_file, request, redirect, flash,url_for
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from concurrent.futures import ThreadPoolExecutor
import io
import tempfile
import os
import rasterio
from osgeo import gdal
from skimage import exposure
import numpy as np
from skimage.segmentation import slic
from rasterio.features import shapes
import joblib
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import pandas as pd
from rasterio.mask import mask
from shapely.geometry import shape
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
import plotly.graph_objects as go
from plotly.offline import plot
import folium
# Just To calculate the amount of memory consumed
from memory_profiler import memory_usage
from collections import defaultdict
import re
from branca.element import Template, MacroElement
from folium.plugins import DualMap

main=Blueprint('main',__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = tempfile.mkdtemp()

# Configura las credenciales para la API de Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
#./pelagic-gist-434920-h5-91777317c681
SERVICE_ACCOUNT_FILE = './pelagic-gist-434920-h5-91777317c681'

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

@main.route('/')
def main_interface():
    return render_template('main_interface.html')

#########################################################################################################
##################################### IMAGE EXTRACTION ##################################################
#########################################################################################################

@main.route('/image_extraction')
def image_extraction():
    return render_template('image_extraction.html')


@main.route('/rasters/<municipality>/<year>')
def download_file(municipality, year):
    try:
        # Paso 1: Encuentra la carpeta principal 'rasters'
        query = f"name='rasters' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        raster_folder_result = drive_service.files().list(q=query, fields="files(id, name)").execute()
        raster_folders = raster_folder_result.get('files', [])

        if not raster_folders:
            return jsonify({"error": "No se encontró la carpeta principal 'rasters'"}), 404

        raster_folder_id = raster_folders[0]['id']
        print(f"Carpeta 'rasters' encontrada: {raster_folder_id}")

        # Paso 2: Encuentra la carpeta del municipio dentro de 'rasters'
        query = f"name='{municipality}' and mimeType='application/vnd.google-apps.folder' and trashed=false and '{raster_folder_id}' in parents"
        folder_result = drive_service.files().list(q=query, fields="files(id, name)").execute()
        folders = folder_result.get('files', [])

        if not folders:
            return jsonify({"error": f"No se encontró la carpeta del municipio: {municipality} dentro de 'rasters'"}), 404

        folder_id = folders[0]['id']
        print(f"Carpeta del municipio {municipality} encontrada: {folder_id}")

        # Paso 3: Encuentra el archivo dentro de la carpeta del municipio
        print(f"Buscando archivo para el año: {year}.tif en la carpeta del municipio: {municipality}")
        query = f"name='{year}' and '{folder_id}' in parents and trashed=false"
        file_result = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = file_result.get('files', [])
        
        print(f"Archivos encontrados: {files}")  # Esto te ayudará a ver los resultados de la búsqueda

        if not files:
            return jsonify({"error": f"No se encontró el archivo para el año: {year} en el municipio: {municipality}"}), 404

        file_id = files[0]['id']
        print(f"Archivo {year}.tif encontrado: {file_id}")

        # Paso 4: Descarga el archivo
        request = drive_service.files().get_media(fileId=file_id)
        file_io = io.BytesIO()
        downloader = MediaIoBaseDownload(file_io, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Descargando {int(status.progress() * 100)}%.")

        file_io.seek(0)
        return send_file(file_io, as_attachment=True, download_name=f"{municipality}_{year}", mimetype='image/tiff')

    except Exception as e:
        print(f"Error: {str(e)}")  # Imprime el error en caso de que ocurra
        return jsonify({"error": str(e)}), 500
#########################################################################################################
##################################### CLASSIFICATION ####################################################
#########################################################################################################

@main.route('/classif', methods=["GET", "POST"])
def classif():

    if request.method == "POST":
        if 'raster' not in request.files:
            flash("No file part")
            return redirect(request.url)

        file = request.files['raster']
        municipality = request.form.get("municipality")
        segments = request.form.get("segments", type=int)

        if file.filename == '':
            flash("No selected file")
            return redirect(request.url)

        if file:
            # Guardar archivo subido
            raster_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(raster_path)
            polygons_classif = clasify(raster_path, municipality,segments)
            return redirect(url_for('main.download_file_classification', filename=os.path.basename(polygons_classif)))
        
    return render_template("classif.html", success=False)

municipality_shapefiles = {
    'Arauca': 'AraucaLC.shp',
    'Arauquita': 'ArauquitaLC.shp',
    'Cravo Norte': 'CravoLC.shp',
    'Fortul': 'FortulLC.shp',
    'Puerto Rondon': 'PuertoLC.shp',
    'Saravena': 'SaravenaLC.shp',
    'Tame': 'TameLC.shp'
}
municipality_dem = {
    'Arauca': 'Arauca_elevation_slope.tif',
    'Arauquita': 'Arauquita_elevation_slope.tif',
    'Cravo Norte': 'Cravo_elevation_slope.tif',
    'Fortul': 'Fortul_elevation_slope.tif',
    'Puerto Rondon': 'Puerto_elevation_slope.tif',
    'Saravena': 'Saravena_elevation_slope.tif',
    'Tame': 'Tame_elevation_slope.tif'
}
def get_file_from_drive(folder_name, file_name):
    """Busca un archivo dentro de una carpeta específica en Google Drive y lo descarga."""
    # Buscar la carpeta "models"
    query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed=false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    folder = results.get("files", [])

    if not folder:
        raise FileNotFoundError(f"No se encontró la carpeta '{folder_name}' en tu unidad de Google Drive.")
    
    folder_id = folder[0]["id"]

    # Buscar el archivo model.joblib dentro de la carpeta models
    query = f"name = '{file_name}' and '{folder_id}' in parents"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    file = results.get("files", [])

    if not file:
        raise FileNotFoundError(f"No se encontró el archivo '{file_name}' en la carpeta '{folder_name}'.")

    file_id = file[0]["id"]

    # Descargar el archivo
    request = drive_service.files().get_media(fileId=file_id)
    file_stream = io.BytesIO()
    downloader = MediaIoBaseDownload(file_stream, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    file_stream.seek(0)
    return file_stream

def clasify(input_path, municipality, segments):
    print("Iniciando procesamiento...")
    name_file = re.search(r'([^/]+)\.tif$', input_path).group(1)
    # Leer las primeras 6 bandas del raster
    mem_before = memory_usage()[0]  ################## MEMORY TRACK
    with rasterio.open(input_path) as src:
        mem_before = memory_usage()[0]  ################## MEMORY TRACK
        band_data = np.stack([src.read(i + 1) for i in range(6)], axis=-1)
        transform, crs = src.transform, src.crs
        # Ajustar intensidades y aplicar segmentación SLIC
        img = exposure.rescale_intensity(band_data)
        segments = slic(img, n_segments=segments, compactness=0.03)
        mem_after = memory_usage()[0]   ############ MEMORY TRACK
        print(f"Memory used by slic {mem_after - mem_before:.2f} MiB")
        mem_before = memory_usage()[0]  ################## MEMORY TRACK
        # Generar geometrías a partir del raster segmentado
        shapes_generator = shapes(segments.astype(np.int32), transform=transform)
        # Crear un GeoDataFrame
        gdf = gpd.GeoDataFrame(({'geometry': shape(geom), 'value': value} for geom, value in shapes_generator),crs=crs)
        # Cargar el área de interés (AOI)
        aoi = gpd.read_file(f'static/aois/{municipality_shapefiles.get(municipality)}')
        # Recortar los polígonos al área de interés
        polygons = gpd.clip(gdf, aoi)
        # Filtrar geometrías vacías
        polygons = polygons[~polygons.is_empty].dropna(subset=['geometry'])
        # Cargar el raster DEM
        dem_raster=rasterio.open(f'static/aois/{municipality_dem.get(municipality)}')
        polygons = polygons.to_crs(dem_raster.crs)
        mem_after = memory_usage()[0]   ############ MEMORY TRACK
        print(f"Memory used by clip and load dem {mem_after - mem_before:.2f} MiB")
        means_spectra_polygon = []
        means_dem_polygon = []


        for idx, polygon in polygons.iterrows():
            geom = [polygon['geometry']]
            try:
                out_image1, _ = mask(src, geom, crop=True)
                out_image2, _ = mask(dem_raster, geom, crop=True)

                # Verificar si hay datos válidos
                if np.isnan(out_image1).all() or np.isnan(out_image2).all():
                    continue  # Saltar este polígono si no tiene datos

                # Calcular la media para cada banda
                means1 = np.nanmean(out_image1, axis=(1, 2))
                means2 = np.nanmean(out_image2, axis=(1, 2))

                means_spectra_polygon.append(dict(polygon_id=polygon['value'], means=means1))
                means_dem_polygon.append(dict(polygon_id=polygon['value'], means=means2))

            except Exception as e:
                print(f"Error al procesar polígono {polygon['value']}: {e}")
                continue  # Saltar en caso de error
            # Convertir a DataFrame
        means_df1 = pd.DataFrame(means_spectra_polygon)
        means_df2 = pd.DataFrame(means_dem_polygon)

        # Asegurar que los índices coincidan
        polygons = polygons.reset_index(drop=True)

        band_names = ['Blue_mean', 'Green_mean', 'Red_mean', 'NIR_mean', 'WIR1_mean', 'WIR2_mean']

        # Convertir la columna 'means' en DataFrame asegurando que cada lista tenga 6 valores
        means_expanded_df = means_df1['means'].apply(
            lambda x: pd.Series(np.pad(x[:6], (0, max(0, 6 - len(x))), constant_values=np.nan))
        )

        # Renombrar las columnas con los nombres de las bandas
        means_expanded_df.columns = band_names

        # Asignar directamente los valores al DataFrame polygons
        polygons[band_names] = means_expanded_df

        # Asignar valores de elevación y pendiente
        band_names = ['Elevation_', 'Slope_mean']
        # Convertir la columna 'means' en DataFrame para facilitar la asignación
        means_expanded = pd.DataFrame(means_df2['means'].to_list(), columns=band_names)
        # Asignar directamente los valores al DataFrame polygons
        polygons[band_names] = means_expanded

        # Seleccionar y renombrar columnas
        gdf_X = polygons[['Blue_mean', 'Green_mean', 'Red_mean', 'NIR_mean', 'WIR1_mean', 'WIR2_mean', 'Elevation_', 'Slope_mean']]
        gdf_X.columns = ['Blue', 'Green', 'Red', 'NIR', 'WIR1', 'WIR2', 'Elevation', 'Slope']

        # Normalizar valores espectrales
        gdf_X[['Blue', 'Green', 'Red', 'NIR', 'WIR1', 'WIR2']] = gdf_X[['Blue', 'Green', 'Red', 'NIR', 'WIR1', 'WIR2']] * 0.0000275 - 0.2

        # Cálculo de índices
        gdf_X['ndvi'] = (gdf_X['NIR'] - gdf_X['Red']) / (gdf_X['NIR'] + gdf_X['Red'])
        gdf_X['rvi'] = gdf_X['WIR1'] / gdf_X['NIR']
        gdf_X['evi'] = 2.5 * (gdf_X['NIR'] - gdf_X['Red']) / (gdf_X['NIR'] + (gdf_X['Red'] * 6) - (gdf_X['Blue'] * 7.5) + 1)
        
        # Normalizar datos
        scaler = StandardScaler()
        gdf_X_standarized = pd.DataFrame(scaler.fit_transform(gdf_X), columns=gdf_X.columns)

        # Convertir de nuevo a GeoDataFrame
        gdf_X_standarized['geometry'] = polygons.geometry
        gdf_X_standarized = gpd.GeoDataFrame(gdf_X_standarized, geometry='geometry', crs=polygons.crs)

        # Cargar el modelo desde el streamsads}
        model = joblib.load(get_file_from_drive("models", "model.joblib"))

        gdf_final = gpd.GeoDataFrame()
        gdf_X = gdf_X_standarized.drop(columns='geometry')
        gdf_X = gdf_X[['Blue', 'Green', 'Red', 'NIR', 'WIR1', 'WIR2', 'ndvi', 'rvi', 'evi', 'Elevation', 'Slope']]
    
        gdf_final['class'] = model.predict(gdf_X)
        gdf_final['geometry'] = polygons.geometry
        gdf_final = gpd.GeoDataFrame(gdf_final)
        # Guardar como GeoJSON
        geojson_filename = os.path.join(RESULT_FOLDER, f"Class_{name_file}.geojson")
        gdf_final.to_file(geojson_filename, driver='GeoJSON')

        mem_after = memory_usage()[0]   ############ MEMORY TRACK
        print(f"Total memory used by the function: {mem_after - mem_before:.2f} MiB")
        
        return geojson_filename

@main.route('/download2/<filename>')
def download_file_classification(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    return send_file(file_path, as_attachment=True)



#########################################################################################################
##################################### Analysis ##########################################################
#########################################################################################################

# Diccionario de colores
class_colors = {  
    'Urban Zones': '#761800',
    'Industry and Comerciall': '#934741',
    'Mining': '#4616d4',
    'Pastures': '#cddc97',
    'Agricultural Areas': '#dbc382',
    'Forest': '#3a6a00',
    'Shrublands and Grassland': '#cafb4d',
    'Little vegetation areas': '#bfc5b9',
    'Continental Wetlands': '#6b5c8c',
    'Continental Waters': '#0127ff'
}



def create_synchronized_maps(gdf1, gdf2, map_id='map_sync'):
    # Convertir a EPSG:4326 si es necesario
    if gdf1.crs.to_string() != "EPSG:4326":
        gdf1 = gdf1.to_crs(epsg=4326)
    if gdf2.crs.to_string() != "EPSG:4326":
        gdf2 = gdf2.to_crs(epsg=4326)

    # Calcular centro del mapa
    center = gdf1.unary_union.centroid.coords[:][0][::-1]

    # Crear mapa dual
    dual_map = DualMap(location=center, zoom_start=12)

    # Unir clases únicas
    all_classes = sorted(set(gdf1['class'].unique()).union(set(gdf2['class'].unique())))

    # Añadir geometrías al primer mapa
    folium.GeoJson(
        gdf1,
        name="LULC 1",
        style_function=lambda feature: {
            'fillColor': class_colors.get(feature['properties']['class'], '#000000'),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['class']),
    ).add_to(dual_map.m1)

    # Añadir geometrías al segundo mapa
    folium.GeoJson(
        gdf2,
        name="LULC 2",
        style_function=lambda feature: {
            'fillColor': class_colors.get(feature['properties']['class'], '#000000'),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['class']),
    ).add_to(dual_map.m2)

    # Crear leyenda con tus colores
    legend_html = """
    <div style='position: fixed; 
         top: 10px; right: 10px; width: 200px; height: auto; 
         z-index:9999; font-size:12px;
         background-color:white; padding:10px; border:2px solid grey;'>
         <b>LU/LC Classes</b><br>
    """
    for cls in all_classes:
        color = class_colors.get(cls, "#000000")
        legend_html += f"<div style='margin-bottom:5px;'><i style='background:{color};width:12px;height:12px;display:inline-block;margin-right:5px;'></i>{cls}</div>"
    legend_html += "</div>"

    # Agregar leyenda al HTML
    dual_map.get_root().html.add_child(folium.Element(legend_html))

    # Guardar mapa
    map_path = os.path.join("static", "map1.html")
    dual_map.save(map_path)

    return map_path


@main.route('/change', methods=['GET', 'POST'])
def change():
    graph_url = None
    sankey_url = None
    map_sync_url = None  

    if request.method == 'POST':
        if 'file1' not in request.files or 'file2' not in request.files:
            return 'No files part'

        file1 = request.files['file1']
        file2 = request.files['file2']

        if file1 and file2:
            upload_folder = UPLOAD_FOLDER
            file1_path = os.path.join(upload_folder, file1.filename)
            file2_path = os.path.join(upload_folder, file2.filename)

            file1.save(file1_path)
            file2.save(file2_path)

            gdf1 = gpd.read_file(file1_path)
            gdf2 = gpd.read_file(file2_path)

            map_sync_url = create_synchronized_maps(gdf1, gdf2, map_id='map_sync')

            if gdf1.crs.to_string() == "EPSG:4326":
                gdf1 = gdf1.to_crs(epsg=32618)

            if gdf2.crs.to_string() == "EPSG:4326":
                gdf2 = gdf2.to_crs(epsg=32618)

            # Bar plot
            gdf1["Area"] = gdf1.geometry.area / 1e6
            gdf2["Area"] = gdf2.geometry.area / 1e6

            area_por_clase1 = gdf1.groupby("class")["Area"].sum().reset_index()
            area_por_clase2 = gdf2.groupby("class")["Area"].sum().reset_index()

            df_area = pd.merge(area_por_clase1, area_por_clase2, on="class", suffixes=('_1992', '_2012'))
            df_area["diff"] = df_area["Area_2012"] - df_area["Area_1992"]
            df_area["Change"] = df_area["diff"].apply(lambda x: "Increase" if x > 0 else "Decrease")
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x="class", y="diff", data=df_area, hue="Change", palette=["#67a9cf", "#ef8a62"])
            plt.axhline(0, color="black", linestyle="--")
            plt.xlabel("Class")
            plt.ylabel("Area (Km²)")
            plt.title("Change in Area")
            plt.xticks(rotation=45, ha="right")
            plt.legend(title="Change", loc="upper left")
            plt.tight_layout()

            graph_path = os.path.join('static', 'images', 'area_change.png')
            plt.savefig(graph_path)
            plt.close()

            graph_url = url_for('static', filename='images/area_change.png')
            
            # Sankey plot
            inter=gpd.overlay(gdf1,gdf2,how='intersection')
            if inter.crs.is_geographic:
                inter = inter.to_crs(epsg=3116)
            inter['area']=inter.geometry.area/1e6
            inter = inter.rename(columns={'class_1': 'Date1','class_2':'Date2'})
            
            label_set = set()
            links = []

            def add_links(inter, col_from, col_to):
                temp = inter.groupby([col_from, col_to])["area"].sum().reset_index()
                for _, row in temp.iterrows():
                    source = f"{col_from}_{row[col_from]}"
                    target = f"{col_to}_{row[col_to]}"
                    value = row["area"]
                    label_set.update([source, target])
                    links.append((source, target, value))

            add_links(inter, "Date1", "Date2")
            
            # Asignar índices
            labels = list(label_set)
            label_to_index = {label: i for i, label in enumerate(labels)}

            source = [label_to_index[s] for s, t, v in links]
            target = [label_to_index[t] for s, t, v in links]
            value = [v for s, t, v in links]

            # Diccionario de colores
            color_dict = {
                "1.1 Urban": "#761800",
                "1.2 Industrial and commercial": "#934741",
                "1.3 Mine, dump and construction": "#4616d4",
                "1.4 Artificial non-agricultural vegetated areas": "#A600CC",
                "2.1 Arable Land": "#e8d610",
                "2.2 Permanent crops": "#F2CCAA",
                "Pastures": "#cddc97",
                "2.4 Shrublands and Grassland": "#dbc382",
                "Forest": "#3a6a00",
                "Shrublands and Grassland": "#cafb4d",
                "3.3 Little Vegetated Areas": "#bfc5b9",
                "4.1 Wetlands": "#6b5c8c",
                "5.1 Water bodies": "#0127ff"
            }
            display_labels = [label.split("_", 1)[1] for label in labels]

            # Colores para los nodos
            colors = []
            for label in labels:
                class_name = label.split("_", 1)[1].strip()
                colors.append(color_dict.get(class_name, "gray"))

            def hex_to_rgba(hex_color, alpha=0.6):
                hex_color = hex_color.lstrip("#")
                r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                return f"rgba({r},{g},{b},{alpha})"

            missing_classes = set()
            for s, t, v in links:
                class_name = s.split("_", 1)[1].strip()
                if class_name not in color_dict:
                    missing_classes.add(class_name)

            link_colors = []
            for s, t, v in links:
                class_name = s.split("_", 1)[1].strip()
                hex_color = color_dict.get(class_name, "#999999")
                link_colors.append(hex_to_rgba(hex_color))
            
            # Crear el Sankey plot
            fig = go.Figure(data=[go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=display_labels,
                    color=colors
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    color=link_colors
                )
            )])

            fig.update_layout(
                title_text="Sankey plot Date 1 and Date 2",
                font_size=12
            )
            # Guardar como HTML
            graph2_path = os.path.join('static', 'images', 'sankey_change.html')
            fig.write_html(graph2_path)
            sankey_url = url_for('static', filename='images/sankey_change.html')
            

    return render_template('change.html', map_url=map_sync_url, graph_url=graph_url, sankey_url=sankey_url)




#########################################################################################################
##################################### Conflict ##########################################################
#########################################################################################################

@main.route('/download3/<filename>')
def download_file_conflict(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Error: File not found", 404

@main.route('/conflict', methods=['GET', 'POST'])
def conflict():
    bar_url=None
    map_url = None
    download_url=None
    if request.method == 'POST':
        file1 = request.files.get('file1')
        classification_type = request.form.get('classification-type')  # Leer selección del usuario
        print(f"Received classification type: {classification_type}")
        if not file1:
            return "Error: No file uploaded", 400

        file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
        file1.save(file1_path)
        name_file=re.search(r"(?<=Class_)(.*?)(?=\.geojson)", file1.filename).group()
        try:
            gdf1 = gpd.read_file(file1_path)
            if gdf1.empty:
                return "Error: No valid shapefile found in the uploaded file", 400
            
            # Seleccionar el shapefile correcto
            if classification_type == "Vocation":
                shapefile_path = os.path.join('static', 'vocation', 'vocation_Arauca.shp')
                column_name = 'Vocation'  # Nombre de la columna en vocación
            elif classification_type == "Ambiental offer":
                shapefile_path = os.path.join('static', 'vocation', 'ambiental_offer.geojson')
                column_name = 'Vocation'  # Nombre de la columna en oferta ambiental
            else:
                return "Error: Invalid classification type", 400

            if not os.path.exists(shapefile_path):
                return f"Error: {classification_type} shapefile not found", 500

            gdf2 = gpd.read_file(shapefile_path)

            # Asegurar que los CRS coincidan
            if gdf1.crs != gdf2.crs:
                gdf1 = gdf1.to_crs(gdf2.crs)

            # Join espacial
            result = gpd.sjoin(gdf1, gdf2[['geometry', column_name]], how='left', predicate='intersects')

            if result is None or result.empty:
                raise ValueError("No valid result from spatial join")

            # Mapeo de clases
            labels_dict = {
                'Agricultural Areas': '24', 'Continental Waters': '51',
                'Continental Wetlands': '41', 'Forest': '31',
                'Industry and Commercial': '12', 'Little vegetation areas': '33',
                'Mining': '13', 'Pastures': '23', 'Shrublands and Grassland': '32',
                'Urban Zones': '11',
            }
            result['Level 2'] = result['class'].replace(labels_dict)

            # Función para asignar conflicto
            def assign_conflict_vocation (row):
                conflict_mapping = {
                    "Agricultural": {"High": ["11", "12", "13", "31", "33", "41", "51"],
                                 "Moderate": ["32", "22"],
                                 "No Conflict": ["21", "23", "24"]},
                    "Livestock": {"High": ["11", "12", "13", "31", "41", "51"],
                                 "Moderate": ["21", "23", "24", "32", "33"],
                                 "No Conflict": ["22"]},
                    "Agroforestry": {"High": ["11", "12", "13", "41", "51"],
                                     "Moderate": ["21", "22", "33"],
                                     "No Conflict": ["23", "24", "31", "32"]},
                    "Forestry": {"High": ["11", "12", "13", "21", "22", "23", "24"],
                                 "Moderate": ["33"],
                                 "No Conflict": ["31", "32", "41", "51"]},
                    "Soil Conservation": {"High": ["11", "12", "13", "41", "51"],
                                               "Moderate": ["21", "22", "33"],
                                               "No Conflict": ["23", "24", "31", "32"]},
                    "Water body": {"High": ["11", "12", "13", "21", "22", "23", "24", "31", "32", "33"],
                                       "No Conflict": ["41", "51"]},
                    "Urban areas": {"High": ["11", "12", "13", "21", "22", "23", "24", "31", "32", "33", "41", "51"]}
                }
                for level, values in conflict_mapping.get(row[column_name], {}).items():
                    if row["Level 2"] in values:
                        return level
                return "Unknown"
            def assign_conflict_enviro (row):
                conflict_mapping = {
                    "Agricultural": {"High": ["11","12","31","51"],
                                 "Moderate": ["13","32","33","41"],
                                 "No Conflict": ["21","23","24","22"]},
                    "Livestock": {"High": ["11","12","31","51"],
                                 "Moderate": ["13","32","33","41"],
                                 "No Conflict": ["21","23","24","22"]},
                    "Agroforestry": {"High": ["11","12","31","51"],
                                     "Moderate": ["13","32","33","41"],
                                     "No Conflict": ["21","23","24","22"]},
                    "Forestry": {"High": ["11","12","13","21","23","24"],
                                 "Moderate": ["33"],
                                 "No Conflict": ["31","32","41","51"]},
                    "Legal Protection Areas": {"High": ["11","12","13","21","23","24","22"],
                                               "Moderate": [],
                                               "No Conflict": ["31","32","33","41","51"]},
                    "Soil Conservation": {"High": ["11","12","13","21","23","24","22"],
                                               "Moderate": ["33"],
                                               "No Conflict": ["31","32","41","51"]},
                    "Water body": {"High": ["11","12","13","21","23","24","22"],
                                       "No Conflict": ["31","32","33","41","51"]},
                    "Urban areas": {"High": ["13","31","32","33"],
                                               "Moderate": ["21","22","23","24","41","51"],
                                               "No Conflict": ["11","12"]}
                }
                for level, values in conflict_mapping.get(row[column_name], {}).items():
                    if row["Level 2"] in values:
                        return level
                return "Unknown"
            # Seleccionar el shapefile correcto
            if classification_type == "Vocation":
                result["Conflict_Level"] = result.apply(assign_conflict_vocation, axis=1)
            elif classification_type == "Ambiental offer":
                result["Conflict_Level"] = result.apply(assign_conflict_enviro, axis=1)


            # Guardar resultado en GeoJSON
            geojson_filename = os.path.join(RESULT_FOLDER, f"Conflict_{name_file}.geojson")
            result.to_file(geojson_filename, driver='GeoJSON')
            
            # Create Map
            m = folium.Map(location=[6.5, -70.8], zoom_start=8.1)
            color_dict = {
                "High": "red",
                "Moderate": "yellow",
                "No Conflict": "green"
            }
            def style_function(feature):
                conflict_level = feature["properties"].get("Conflict_Level", "No Conflict")
                return {
                    "fillColor": color_dict.get(conflict_level, "gray"),
                    "color": "black",
                    "weight": 1,
                    "fillOpacity": 1,
                }
            folium.GeoJson(
                result.to_json(),  # ✅ Convierte correctamente a GeoJSON
                style_function=style_function,
                name="Conflict"
            ).add_to(m)
            
            legend_html = """
            <div style='position: fixed; 
                top: 10px; right: 10px; width: 120px; height: 100px; 
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color: white;
                padding: 10px;'>
                <strong>Conflict</strong><br>
                <i style="background:red; width:10px; height:10px; display:inline-block;"></i> High<br>
                <i style="background:yellow; width:10px; height:10px; display:inline-block;"></i> Moderate<br>
                <i style="background:green; width:10px; height:10px; display:inline-block;"></i> No Conflict
            </div>
            """
            # Bar plot
            gdf=result
            if gdf.crs.is_geographic:
                gdf = gdf.to_crs(epsg=3116)
            gdf['area'] = gdf.geometry.area / 1e6
            # The code `area_conf` is not valid Python syntax. It seems like it might be a placeholder
            # or a comment in the code. It does not perform any specific action or operation in
            # Python.
            area_conf = gdf.groupby('Conflict_Level')['area'].sum().reset_index()
            area_conf = area_conf.sort_values('area', ascending=False)
            def asignar_color(area):
                if area < area_conf['area'].quantile(0.33):
                    return '#ffeda0'  # amarillo claro
                elif area < area_conf['area'].quantile(0.66):
                    return '#feb24c'  # naranja
                else:
                    return '#f03b20'  # rojo fuerte

            area_conf['color'] = area_conf['area'].apply(asignar_color)
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=area_conf['Conflict_Level'],
                y=area_conf['area'],
                marker_color=area_conf['color'],
                text=area_conf['area'].round(2),
                textposition='inside'
            ))

            fig.update_layout(
                yaxis_title='Area (Km²)',
                xaxis_tickangle=-45,
                margin=dict(t=50, b=150),
                yaxis=dict(title_font=dict(size=14)),
                xaxis=dict(title_font=dict(size=14)),
                uniformtext_minsize=8,
                uniformtext_mode='hide',
            )

            bar_path = os.path.join('static', 'images', 'bar_conflict.html')
            fig.write_html(bar_path)
            
            bar_url = url_for('static', filename='images/bar_conflict.html')
            m.get_root().html.add_child(folium.Element(legend_html))
            map_path = os.path.join('static', 'images', 'map_conflict.html')
            m.save(map_path)
            map_url = url_for('static', filename='images/map_conflict.html')
            download_url = url_for('main.download_file_conflict', filename=os.path.basename(geojson_filename))
            
            return render_template('conflict.html', conflict_map_url=map_url, download_url=download_url,bar_url=bar_url)

        except Exception as e:
            print(f"Error processing shapefiles: {e}")
            return "Error processing the shapefiles", 500

    return render_template('conflict.html')

#########################################################################################################
##################################### Vizualization #####################################################
#########################################################################################################
@main.route('/analysis')
def analysis():
    return render_template('analysis.html')

@main.route('/soil_use_map')
def soil_use_map():
    map_html_path = 'static/shape1992.html'

    return render_template('soil_use_map.html', map_path=map_html_path)

@main.route('/compare')
def compare():
    return render_template('compare.html')

