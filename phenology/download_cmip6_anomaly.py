import itertools
import os

import ee
import geemap

ee.Authenticate()
ee.Initialize(project="ee-cjc378")

start_date = '2020-01-01'
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

#UNUSED: 'GFDL-CM4', 'NorESM2-MM', 'CESM2', 'CESM2-WACCM', 'IITM-ESM', 'TaiESM1',
#
models = ['ACCESS-CM2']#, 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CMCC-CM2-SR5', 
        #   'CMCC-ESM2','CNRM-CM6-1', 'CNRM-ESM2-1', 'CanESM5', 'EC-Earth3', 
        #   'EC-Earth3-Veg-LR', 'FGOALS-g3', 'GFDL-ESM4', 'GISS-E2-1-G', 
        #   'HadGEM3-GC31-LL', 'HadGEM3-GC31-MM', 'INM-CM4-8', 'INM-CM5-0', 
        #   'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR', 
        #   'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM',   'UKESM1-0-LL']
scenarios = ['ssp245']#, 'ssp585']
week_starts = list(range(1, 366, 7))
week_ends = list(range(7, 360, 7)) + [365] # Last week goes until end of year.
years = list(range(2086, 2101))

# Create the folder if it doesn't exist
output_folder = f'../data/cmip6/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
out_dir = os.path.join(os.getcwd(), output_folder)

for config in itertools.product(models, scenarios, years, zip(week_starts, week_ends)):
    model, scenario, year, (week_start, week_end) = config
    if os.path.exists(f'{output_folder}{model}_{scenario}_{year}_{week_start}_{week_end}.tif'):
        print(f'Skipping {model} {scenario} {year} week {week_start}-{week_end} because file already exists.')
        continue
    # Prepare CMIP6 Predictions for future conditions
    cmip6_future = (ee.ImageCollection(f"NASA/GDDP-CMIP6")
                        .filterBounds(region)
                        .filter(ee.Filter.eq('year', year))
                        .filter(ee.Filter.dayOfYear(week_start, week_end))
                        .filter(ee.Filter.eq('model', model))
                        .filter(ee.Filter.eq('scenario', scenario))
                        .select(['tasmax', 'tasmin', 'tas'])
                        .mean())
    
    # Prepare CMIP6 estimates for historical conditions
    cmip6_historical = (ee.ImageCollection(f"NASA/GDDP-CMIP6")
                            .filterBounds(region)
                            .filter(ee.Filter.gte('year', 2000))
                            .filter(ee.Filter.dayOfYear(week_start, week_end))
                            .filter(ee.Filter.eq('model', model))
                            .filter(ee.Filter.eq('scenario', 'historical'))
                            .select(['tasmax', 'tasmin', 'tas'])
                            .mean())
    
    cmip6_anomaly = (cmip6_future.subtract(cmip6_historical)
                        .set('year', year)
                        .set('week_start', week_start)
                        .set('week_end', week_end)
                        .set('model', model)
                        .set('scenario', scenario)
                        .set('system:index:', f'{model}_{scenario}_{year}_{week_start}_{week_end}'))

    def shrink(image):
        image = ee.Image(image).select(['tasmax', 'tasmin', 'tas']).clip(region)
        # Convert from Kelvin to Celsius, then scale to 16-bit unsigned integer 
        # range (0-65535) for storage.
        image = image.subtract(-60).divide(120).multiply(65535).uint16()
        return (ee.Image(image.copyProperties(image))
                .setDefaultProjection('EPSG:4326', [0.25,0,-180,0,-0.25,90]))


    cmip6_anomaly = ee.Image(shrink(cmip6_anomaly))


    # download images
    #geemap.download_ee_image_collection(cmip6_anomaly, out_dir, region=region)
    geemap.download_ee_image(
        cmip6_anomaly, 
        f'../data/cmip6/{model}_{scenario}_{year}_{week_start}_{week_end}.tif',
        region, 'EPSG:4326', [0.25,0,-180,0,-0.25,90], dtype='float64')