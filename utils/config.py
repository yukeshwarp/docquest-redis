import os
from dotenv import load_dotenv

load_dotenv()

azure_endpoint = "https://uswest3daniel.openai.azure.com"
api_key = "fcb2ce5dc289487fad0f6674a0b35312"
api_version ="2024-10-01-preview"
model = "GPT-4Omni"
azure_function_url ='https://doc2pdf.azurewebsites.net/api/HttpTrigger1'
redis_host ="yuktestredis.redis.cache.windows.net"
redis_pass = "VBhswgzkLiRpsHVUf4XEI2uGmidT94VhuAzCaB2tVjs="

# AZURE_ENDPOINT="https://uswest3daniel.openai.azure.com"
# API_KEY="fcb2ce5dc289487fad0f6674a0b35312"
# API_VERSION="2024-10-01-preview"
# MODEL="GPT-4Omni"
# AZURE_FUNCTION_URL = 'https://doc2pdf.azurewebsites.net/api/HttpTrigger1'
# HOST_NAME = "yuktestredis.redis.cache.windows.net"
# PASSWORD = "VBhswgzkLiRpsHVUf4XEI2uGmidT94VhuAzCaB2tVjs="