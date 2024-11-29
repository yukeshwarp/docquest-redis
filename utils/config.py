import os
from dotenv import load_dotenv

load_dotenv()

azure_endpoint = "https://uswest3daniel.openai.azure.com"
api_key = "fcb2ce5dc289487fad0f6674a0b35312"
api_version = "2024-10-01-preview"
model = "GPT-4Omni"
azure_function_url = 'https://doc2pdf.azurewebsites.net/api/HttpTrigger1'
redis_host = "yuktestredis.redis.cache.windows.net"
redis_pass = "VBhswgzkLiRpsHVUf4XEI2uGmidT94VhuAzCaB2tVjs="


# azure_endpoint = os.getenv("AZURE_ENDPOINT")
# api_key = os.getenv("API_KEY")
# api_version = os.getenv("API_VERSION")
# model = os.getenv("MODEL")
# azure_function_url = os.getenv("AZURE_FUNCTION_URL")
# redis_host = os.getenv("HOST_NAME")
# redis_pass = os.getenv("PASSWORD")