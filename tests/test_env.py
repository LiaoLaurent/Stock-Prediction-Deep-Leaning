import os
import dotenv

dotenv.load_dotenv()

print(os.getenv("START_DATE"))
print(os.getenv("END_DATE"))
