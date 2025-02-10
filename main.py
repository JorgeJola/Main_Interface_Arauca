from flask import Blueprint, Flask,render_template, jsonify, send_file, request, redirect, flash,url_for
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io
import tempfile
import os
import rasterio
from osgeo import gdal
from skimage import exposure
import numpy as np
from skimage.segmentation import slic
from rasterio.features import shapes
import zipfile
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
import folium

main=Blueprint('main',__name__)
UPLOAD_FOLDER = tempfile.mkdtemp()
RESULT_FOLDER = tempfile.mkdtemp()

# Configura las credenciales para la API de Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
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
            processed_raster_path = process_raster(raster_path, municipality)
            segmented_raster_path = segment_raster(processed_raster_path, municipality,segments)
            polygons_path = generate_shapefile(segmented_raster_path, municipality)
            polygons_bands = extract_bands(polygons_path, processed_raster_path, municipality)
            polygons_classif = apply_model(polygons_path, polygons_bands, municipality)
            return redirect(url_for('main.download_file_classification', filename=os.path.basename(polygons_classif)))
        
    return render_template("classif.html", success=False)

def process_raster(input_path, municipality):
    output_path = os.path.join(RESULT_FOLDER, f"Cleaned_Raster_{municipality}.tif")

    with rasterio.open(input_path) as multiband_raster:
        # Copiar metadatos y ajustar para el archivo de salida
        new_metadata = multiband_raster.meta
        new_metadata.update(count=6)

        with rasterio.open(output_path, 'w', **new_metadata) as dst:
            for i in range(1, 7):  # Procesar bandas 1 a 6
                band = multiband_raster.read(i)  # Leer banda individualmente
                dst.write(band, indexes=i)  # Escribir banda individualmente

    return output_path


def segment_raster(input_path, municipality, segments):
    print('Empezamos')
    with rasterio.open(input_path) as src:
        nbands = src.count
        width = src.width
        height = src.height
        
        # Preasignar un arreglo para las bandas
        band_data = np.empty((height, width, nbands), dtype=src.dtypes[0])
        
        # Leer las bandas directamente en el arreglo preasignado
        for i in range(nbands):
            band = src.read(i + 1, window=rasterio.windows.Window(0, 0, width, height))
            band_data[:, :, i] = band  # Asignar al arreglo preasignado
            del band

        # Ajustar intensidades y realizar segmentación
        img = exposure.rescale_intensity(band_data)

        # Usar el valor de segmentos proporcionado por el usuario
        segments = slic(img, n_segments=segments, compactness=0.03)

        # Guardar el resultado de la segmentación
        output_path = os.path.join(RESULT_FOLDER, f"Segmented_Raster_{municipality}.tif")

        with rasterio.open(input_path) as src:
            profile = src.profile
            profile.update(dtype=rasterio.float32, count=1)

            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(segments.astype(rasterio.float32), 1)
        print('Terminamos segmentacion')

    return output_path

municipality_shapefiles = {
    'Arauca': 'AraucaLC.shp',
    'Arauquita': 'ArauquitaLC.shp',
    'Cravo Norte': 'CravoLC.shp',
    'Fortul': 'FortulLC.shp',
    'Puerto Rondon': 'PuertoLC.shp',
    'Saravena': 'SaravenaLC.shp',
    'Tame': 'TameLC.shp'
}

def generate_shapefile(segmented_raster_path, municipality):
    # Abrir el raster segmentado
    with rasterio.open(segmented_raster_path) as segmented_raster:
        band = segmented_raster.read(1)  # Leer la primera banda
        transform = segmented_raster.transform  # Obtener la transformación geográfica

        # Generar geometrías y valores a partir del raster
        shapes_generator = shapes(band, transform=transform)
        geometries = []
        values = []

        for geom, value in shapes_generator:
            geometries.append(shape(geom))  # Convertir geometría a objeto Shapely
            values.append(value)  # Guardar valor asociado

    # Crear un GeoDataFrame
    gdf = gpd.GeoDataFrame({'geometry': geometries, 'value': values})
    gdf.set_crs(segmented_raster.crs, allow_override=True, inplace=True)

    # Leer el archivo de área de interés (AOI) para el municipio
    aoi = gpd.read_file(f'static/aois/{municipality_shapefiles.get(municipality)}')

    # Guardar el GeoDataFrame como shapefile
    polygons = gpd.clip(gdf, aoi)
    output_path = os.path.join(RESULT_FOLDER, f"Polygons_{municipality}.shp")

    # Guardar los archivos .shp, .shx, .dbf, etc.
    polygons.to_file(f"{output_path}")
    print('Terminamos shapefile')
    return output_path
municipality_dem = {
    'Arauca': 'Arauca_elevation_slope.tif',
    'Arauquita': 'Arauquita_elevation_slope.tif',
    'Cravo Norte': 'Cravo_elevation_slope.tif',
    'Fortul': 'Fortul_elevation_slope.tif',
    'Puerto Rondon': 'Puerto_elevation_slope.tif',
    'Saravena': 'Saravena_elevation_slope.tif',
    'Tame': 'Tame_elevation_slope.tif'
}
def extract_bands(polygons_path, input_path,municipality):
    dem_raster=rasterio.open(f'static/aois/{municipality_dem.get(municipality)}')
    newbands_raster=rasterio.open(input_path)
    polygons=gpd.read_file(polygons_path)

    polygons = polygons.to_crs(dem_raster.crs)

    means_spectra_polygon = []
    means_dem_polygon = []

    for _, polygon in polygons.iterrows():
        # Extract the polygon geometry
        geom = [polygon['geometry']]
        # Mask the raster with the polygon
        out_image1, out_transform1 = mask(newbands_raster, geom, crop=True)
        out_image2, out_transform2 = mask(dem_raster, geom, crop=True)
        # Calculate the mean for each band in the masked area
        means1 = np.nanmean(out_image1, axis=(1, 2)) 
        means2 = np.nanmean(out_image2, axis=(1, 2)) 
        # Store the means and associated polygon id (or other attributes)
        means_spectra_polygon.append(dict(polygon_id=polygon['value'], means=means1))
        means_dem_polygon.append(dict(polygon_id=polygon['value'], means=means2))

    means_df1 = pd.DataFrame(means_spectra_polygon)
    means_df2 = pd.DataFrame(means_dem_polygon)

    for i, band_mean in enumerate(means_df1['means']):
        # Create a new column for each band
        for j, mean in enumerate(band_mean):
            if j==0:
                band_name = f"Blue_mean"
            elif j==1:
                band_name = f"Green_mean"
            elif j==2:
                band_name = f"Red_mean"
            elif j==3:
                band_name = f"NIR_mean"
            elif j==4:
                band_name = f"WIR1_mean"
            elif j==5:
                band_name = f"WIR2_mean"
            polygons.at[i, band_name] = mean

    for i, band_mean in enumerate(means_df2['means']):
        # Create a new column for each band
        for j, mean in enumerate(band_mean):
            if j==0:
                band_name = f"Elevation_"
            elif j==1:
                band_name = f"Slope_mean"
            polygons.at[i, band_name] = mean

    gdf_X=polygons[['Blue_mean', 'Green_mean', 'Red_mean', 'NIR_mean',
        'WIR1_mean', 'WIR2_mean', 'Elevation_', 'Slope_mean']]
    gdf_X=gdf_X.set_axis(['Blue','Green','Red','NIR','WIR1','WIR2','Elevation','Slope'],axis=1)
    gdf_X[['Blue','Green','Red','NIR','WIR1','WIR2']]=gdf_X[['Blue','Green','Red','NIR','WIR1','WIR2']]*0.0000275-0.2
    gdf_X['ndvi']=(gdf_X['NIR']-gdf_X['Red'])/(gdf_X['NIR']+gdf_X['Red'])
    #gdf_X['ndbi']=(gdf_X['WIR1']-gdf_X['NIR'])/(gdf_X['WIR1']+gdf_X['NIR'])
    gdf_X['rvi']= gdf_X['WIR1']/gdf_X['NIR']
    #gdf_X['dvi']=gdf_X['WIR1']-gdf_X['NIR']
    gdf_X['evi']=2.5 * (gdf_X['NIR'] - gdf_X['Red']) / (gdf_X['NIR'] + (gdf_X['Red'] * 6) - (gdf_X['Blue'] * 7.5) + 1)
    scaler=StandardScaler()
    gdf_X_standarized=pd.DataFrame(scaler.fit_transform(gdf_X),columns=gdf_X.columns)
    gdf_X_standarized['geometry']=polygons.geometry
    gdf_X_standarized=gpd.GeoDataFrame(gdf_X_standarized)

    output_path = os.path.join(RESULT_FOLDER, f"gdf_X_{municipality}.shp")
    gdf_X_standarized.to_file(f"{output_path}")

    #zip_filename = f"{shapefile_base}.zip"
    #with zipfile.ZipFile(zip_filename, 'w') as zipf:
    #    for ext in ['.shp', '.shx', '.dbf', '.prj']:
    #        file_path = f"{shapefile_base}{ext}"
    #        if os.path.exists(file_path):
    #            zipf.write(file_path, os.path.basename(file_path))
    return output_path

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

def apply_model(polygons_path, polygons_band, municipality):
    # Obtener el archivo model.joblib desde Google Drive
    model_stream = get_file_from_drive("models", "model.joblib")
    
    # Cargar el modelo desde el stream
    model = joblib.load(model_stream)

    # Cargar y procesar los datos
    polygons = gpd.read_file(polygons_path)
    gdf_X = gpd.read_file(polygons_band)
    gdf_final = gpd.GeoDataFrame()
    
    gdf_X = gdf_X.drop(columns='geometry')
    gdf_X = gdf_X[['Blue', 'Green', 'Red', 'NIR', 'WIR1', 'WIR2', 'ndvi', 'rvi', 'evi', 'Elevation', 'Slope']]
    
    gdf_final['class'] = model.predict(gdf_X)
    gdf_final['geometry'] = polygons.geometry
    gdf_final = gpd.GeoDataFrame(gdf_final)

    shapefile_base = os.path.join(RESULT_FOLDER, f"Class_{municipality}")
    gdf_final.to_file(f"{shapefile_base}.shp")

    # Crear archivo ZIP con el shapefile
    zip_filename = f"{shapefile_base}.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for ext in ['.shp', '.shx', '.dbf', '.prj']:
            file_path = f"{shapefile_base}{ext}"
            if os.path.exists(file_path):
                zipf.write(file_path, os.path.basename(file_path))

    return zip_filename

@main.route('/download2/<filename>')
def download_file_classification(filename):
    # Construir la ruta del archivo .zip
    file_path = os.path.join(RESULT_FOLDER, filename)
    
    # Enviar el archivo .zip como una descarga
    return send_file(file_path, as_attachment=True)


#########################################################################################################
##################################### Analysis ##########################################################
#########################################################################################################
def extract_shapefile(zip_path, extract_folder):
    print(f"Extracting zip: {zip_path} to {extract_folder}")  # Depuración
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    # Buscar archivos .shp en subdirectorios, ignorando archivos no válidos como '._'
    shapefiles = []
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            # Ignorar archivos con prefijo '._' o archivos no shapefiles
            if file.endswith('.shp') and not file.startswith('._'):
                shapefiles.append(os.path.join(root, file))
    
    print(f"Shapefiles found: {shapefiles}")  # Depuración
    return shapefiles[0] if shapefiles else None

class_colors = {  
    'Urban Zones': '#761800',
    'Industry and Comerciall': '#934741',
    'Mining': '#4616d4',
    'Pastures': '#e8d610',
    'Pastures': '#cddc97',
    'Agricultural Areas': '#dbc382',
    'Forest': '#3a6a00',
    'Shrublands and Grassland': '#cafb4d',
    'Little vegetation areas': '#bfc5b9',
    'Continental Wetlands': '#6b5c8c',
    'Continental Waters': '#0127ff'
}

def create_folium_map(gdf, map_id):
    gdf = gdf.to_crs(epsg=4326)
    # Crear un mapa base centrado en el centro de la geometría
    m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=10)
    
    for _, row in gdf.iterrows():
        # Asignar el color correspondiente de la paleta según la clase
        color = class_colors.get(row['class'], '#808080')  # color predeterminado si no se encuentra en la paleta
        folium.GeoJson(
            row['geometry'],
            style_function=lambda feature, color=color: {
                'fillColor': color,
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.8
            }
        ).add_to(m)
    
    # Guardar el mapa como archivo HTML
    map_path = os.path.join('static', f'{map_id}.html')
    m.save(map_path)
    return map_path

@main.route('/change', methods=['GET', 'POST'])
def change():
    graph_url = None
    map1_url = None 
    map2_url = None 
    if request.method == 'POST':
        # Verificar que los archivos se recibieron
        if 'file1' not in request.files or 'file2' not in request.files:
            return 'No files part'

        file1 = request.files['file1']
        file2 = request.files['file2']

        # Verificar si los archivos fueron subidos correctamente
        if file1 and file2:
            upload_folder = UPLOAD_FOLDER
            file1_path = os.path.join(upload_folder, file1.filename)
            file2_path = os.path.join(upload_folder, file2.filename)
            print(f"Saving file1: {file1.filename} to {file1_path}")  # Depuración
            print(f"Saving file2: {file2.filename} to {file2_path}")  # Depuración

            file1.save(file1_path)
            file2.save(file2_path)

            # Extraer shapefiles
            file1_shapefile = extract_shapefile(file1_path, os.path.join(upload_folder, 'file1'))
            file2_shapefile = extract_shapefile(file2_path, os.path.join(upload_folder, 'file2'))

            # Verificar si se extrajeron shapefiles correctamente
            if not file1_shapefile or not file2_shapefile:
                return "Error: No shapefiles found in the uploaded ZIP files."

            gdf1 = gpd.read_file(os.path.join(upload_folder, 'file1', file1_shapefile))
            gdf2 = gpd.read_file(os.path.join(upload_folder, 'file2', file2_shapefile))

            map1 = create_folium_map(gdf1, 'map1')
            map2 = create_folium_map(gdf2, 'map2')

            # Verificar CRS y transformar si es necesario
            if gdf1.crs.to_string() == "EPSG:4326":
                gdf1 = gdf1.to_crs(epsg=32618)

            if gdf2.crs.to_string() == "EPSG:4326":
                gdf2 = gdf2.to_crs(epsg=32618)

            # Calcular el área
            gdf1["Area"] = gdf1.geometry.area / 1000000
            gdf2["Area"] = gdf2.geometry.area / 1000000

            area_por_clase1 = gdf1.groupby("class")["Area"].sum().reset_index()
            area_por_clase2 = gdf2.groupby("class")["Area"].sum().reset_index()

            df_area = pd.merge(area_por_clase1, area_por_clase2, on="class", suffixes=('_1992', '_2012'))
            df_area["diff"] = df_area["Area_2012"] - df_area["Area_1992"]
            df_area["Change"] = df_area["diff"].apply(lambda x: "Increase" if x > 0 else "Decrease")

            # Graficar el cambio de área
            plt.figure(figsize=(10, 6))
            sns.barplot(x="class", y="diff", data=df_area, hue="Change", palette=["red", "green"])
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

            # Log de la ruta del gráfico
            print(f"Graph saved at: {graph_path}")

            # Generar la URL para el gráfico
            graph_url = url_for('static', filename='images/area_change.png')
            map1_url = url_for('static', filename='map1.html')
            map2_url = url_for('static', filename='map2.html')

    return render_template('change.html', map1_url=map1_url, map2_url=map2_url, graph_url=graph_url)


#########################################################################################################
##################################### Conflict ##########################################################
#########################################################################################################
def extract_shapefile_conflict(zip_path, extract_folder):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
        shapefiles = [
            os.path.join(root, file)
            for root, _, files in os.walk(extract_folder)
            for file in files if file.endswith('.shp') and not file.startswith('._')
        ]
        return shapefiles[0] if shapefiles else None
    except Exception as e:
        print(f"Error extracting shapefile: {e}")
        return None

@main.route('/download3/<filename>')
def download_file_conflict(filename):
    file_path = os.path.join(RESULT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "Error: File not found", 404

@main.route('/conflict', methods=['GET', 'POST'])
def conflict():
    if request.method == 'POST':
        file1 = request.files.get('file1')
        if not file1:
            return "Error: No file uploaded", 400

        file1_path = os.path.join(UPLOAD_FOLDER, file1.filename)
        file1.save(file1_path)

        extract_folder = os.path.join(UPLOAD_FOLDER, 'file1')
        os.makedirs(extract_folder, exist_ok=True)
        shapefile1 = extract_shapefile_conflict(file1_path, extract_folder)

        if not shapefile1:
            return "Error: No valid shapefile found in the uploaded file", 400

        try:
            # Cargar los shapefiles
            gdf1 = gpd.read_file(shapefile1)
            gdf2 = gpd.read_file(
                os.path.join('static', 'vocation', 'vocation_Arauca.shp')
            )

            # Asegurarse de que los CRS coinciden
            if gdf1.crs != gdf2.crs:
                gdf1 = gdf1.to_crs(gdf2.crs)

            # Realizar el join espacial
            result = gpd.sjoin(
                gdf1, gdf2[['geometry', 'Vocacion']], how='left', predicate='intersects'
            )

            # Verificar que 'result' contiene datos válidos
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

            # Función para asignar el nivel de conflicto
            def assign_conflict(row):
                conflict_mapping = {
                    "Agrícola": {"High": ["11", "12", "13", "31", "33", "41", "51"],
                                 "Moderate": ["32", "22"],
                                 "No Conflict": ["21", "23", "24"]},
                    "Ganadera": {"High": ["11", "12", "13", "31", "41", "51"],
                                 "Moderate": ["21", "23", "24", "32", "33"],
                                 "No Conflict": ["22"]},
                    "Agroforestal": {"High": ["11", "12", "13", "41", "51"],
                                     "Moderate": ["21", "22", "33"],
                                     "No Conflict": ["23", "24", "31", "32"]},
                    "Forestal": {"High": ["11", "12", "13", "21", "22", "23", "24"],
                                 "Moderate": ["33"],
                                 "No Conflict": ["31", "32", "41", "51"]},
                    "Conservación de Suelos": {"High": ["11", "12", "13", "41", "51"],
                                               "Moderate": ["21", "22", "33"],
                                               "No Conflict": ["23", "24", "31", "32"]},
                    "Cuerpo de agua": {"High": ["11", "12", "13", "21", "22", "23", "24", "31", "32", "33"],
                                       "No Conflict": ["41", "51"]},
                    "Zonas urbanas": {"High": ["11", "12", "13", "21", "22", "23", "24", "31", "32", "33", "41", "51"]},
                }
                for level, values in conflict_mapping.get(row["Vocacion"], {}).items():
                    if row["Level 2"] in values:
                        return level
                return "Unknown"

            result["Conflict_Level"] = result.apply(assign_conflict, axis=1)

            # Guardar el resultado como shapefile
            output_path = os.path.join(RESULT_FOLDER, 'conflict')
            result.to_file(f"{output_path}.shp")

            # Comprimir el archivo de salida
            zip_filename = f"{output_path}.zip"
            with zipfile.ZipFile(zip_filename, 'w') as zipf:
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    file_path = f"{output_path}{ext}"
                    if os.path.exists(file_path):
                        zipf.write(file_path, os.path.basename(file_path))

            # Redirigir a la descarga
            return redirect(url_for('main.download_file_conflict', filename=os.path.basename(zip_filename)))

        except Exception as e:
            print(f"Error processing shapefiles: {e}")
            return "Error processing the shapefiles", 500

    return render_template('conflict.html')

#########################################################################################################
##################################### Vizualization ##########################################################
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
