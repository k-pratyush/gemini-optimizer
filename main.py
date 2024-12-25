from opro_optimizer.settings import settings
from opro_optimizer.utils import call_palm_server_from_cloud

call_palm_server_from_cloud("hello world", model="gemini-1.5-flash-002", max_decode_steps=20, temperature=0.8)