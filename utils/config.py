import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file


azure_endpoint = os.getenv("AZURE_ENDPOINT")
api_key = os.getenv("API_KEY")
api_version = os.getenv("API_VERSION")
model = os.getenv("MODEL")
azure_function_url = os.getenv("AZURE_FUNCTION_URL")
redis_host = os.getenv("HOST_NAME")
redis_pass = os.getenv("PASSWORD")
azure_blob_connection_string = os.getenv("BLOB_CONNECTION_STRING")
azure_container_name = os.getenv("BLOB_CONTAINER_NAME")
