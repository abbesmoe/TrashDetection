import functions as func

# Setting global variables
# Lists to store trash categories for the search page to display
TRASH_LIST = ["Bottle", "Pop tab", "Can", "Bottle cap", "Cigarette", "Cup", "Lid", "Other", "Plastic bag + wrapper", "Straw"]
SELECTED_TRASH_LIST = []

# Lists identifying which trash categories are recyclable and which are not
RECYCLABLES = ["Bottle", "Bottle cap", "Can", "Plastic bag + wrapper", "Pop tab"]
NON_RECYCLABLES = ["Cigarette","Cup", "Lid","Other","Straw"]

# Lists to store the search page table headers and rows
SEARCH_TABLE_HEADERS = ["Images","Quantity"]
SEARCH_TABLE_ROWS = []

# Additional variables for the search page
QUANTITY_FILTER = ""               # Quantity value provided in search page
QUANTITY_TYPE_FILTER = ""           # Quantity filter type (>,<,=)
INTERSECTION_FILTER = "False"      # Intersection filter
RECYCLABLE_FILTER = "False"
NON_RECYCLABLE_FILTER = "False"

IMAGES_DATA = {"Images":[]}

# For files upload in upload page
ALLOWED_EXTENSIONS = set(['jpg','png','jpeg','img','gif','mp4'])

# Files
UPLOAD_PATH = 'static/uploads/'
ANNOTATED_IMAGES_PATH = "static/annotated_images/"
JSON_DATA_FILE = "data/data.json"
MAP_FILE = "detector/taco_config/map_10.csv"
TACO_DIR = "data"
SAMPLE_IMG = 'sample.JPG'
SAMPLE_ANN_IMG = 'output_sample.JPG'

MODEL = func.load_model()