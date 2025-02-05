from flask import Blueprint, Flask,render_template, jsonify, send_file
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io

main=Blueprint('main',__name__)

# Configura las credenciales para la API de Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = './pelagic-gist-434920-h5-91777317c681'

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

@main.route('/')
def main_interface():
    return render_template('main_interface.html')

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
