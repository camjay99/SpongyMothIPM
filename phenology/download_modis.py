import os

import ee
import geemap
from numpy import var

ee.Authenticate()
ee.Initialize(project="ee-cjc378")

start_date = '2000-01-01'
end_date = '2026-01-01'

# Get region to be downloaded (either specify or create with job arrays)
region = ee.Geometry.Polygon(
        [[[-93.39494888939744, 48.80312064746307],
          [-93.52678482689744, 43.44879339811958],
          [-88.78069107689744, 39.87165812618535],
          [-83.63908951439744, 36.239689868809805],
          [-76.43205826439744, 35.813204962789506],
          [-75.50920670189744, 36.452065179561416],
          [-73.34490006127243, 40.56792341674124],
          [-69.75237076439744, 41.50432157200159],
          [-66.63808951311132, 44.68870274464322],
          [-65.78434766221059, 43.121890293567795],
          [-58.577316412210585, 45.81813291738611],
          [-60.379074224710585, 47.23901607826479],
          [-61.741378912210585, 46.33636098978076],
          [-64.59782422471059, 47.29865507073352],
          [-63.674972662210585, 49.05663917284978],
          [-66.17985547471059, 49.54379495108128],
          [-72.70964078275824, 50.948680153415054],
          [-80.02609348194582, 51.52064726455817],
          [-88.22233609525824, 51.197189028268944],
          [-93.27604703275824, 49.91297330236462]]])

# Prepare MODIS Imagery
def correct_sos(image):
    # MODIS Landcover Dynamics product (MCD12Q2) provides the date 
    # of greenup as the number of days since January 1, 1970.
    # This corrects the date to be the number of days since the start of the year.
    # Note that the first day of the year is 0.
    base = ee.Date.fromYMD(1970, 1, 1)
    im_date = ee.Date(image.get('system:time_start'))
    start_of_year = im_date.difference(base, "day")
    return image.subtract(start_of_year).rename('SOS').copyProperties(image)

phenology = (ee.ImageCollection('MODIS/061/MCD12Q2')
                .filterDate(start_date, end_date)
                .filterBounds(region)
                .select('Greenup_1')
                .map(correct_sos))

# Create the folder if it doesn't exist
output_folder = './Test'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
out_dir = os.path.join(os.getcwd(), output_folder)

# download images
geemap.download_ee_image_collection(phenology, out_dir, region=region, crs='EPSG:4326', scale=500)