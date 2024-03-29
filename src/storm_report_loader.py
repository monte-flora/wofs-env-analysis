import pandas as pd 
import xarray as xr 
from datetime import datetime, timedelta
import numpy as np 
from os.path import join
import pyresample
import itertools 
from scipy.ndimage import maximum_filter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class StormReportLoader:
    """
    StormReportLoader loads CSV files containing data about timing and locations of
    storm report data (e.g., hail, tornadoes). The code is designed for 
    the re-formatted Storm Data generated by wofs_ml_severe.data_pipeline.StormReportDownloader
    and the raw CSV files from the Iowa State repo. 

    Attributes:
    -------------------------
        reports_path : path-like, str
            Path to the CSV file with the report data. 
            
        report_type : 'NOAA' or 'IOWA'
            If 'NOAA', the report dataset is structured like 
            the NOAA Storm Event CSV file. 
            IF 'IOWA', the report dataset is structured like 
            the Iowa State CSV files. 
    
        initial_time, string (format = YYYYMMDDHHmm)
            The beginning date and time of a forecast period.

        forecast_length, integer
            Forecast length (in minutes) (default=30)

        err_window, integer 
            Allowable reporting error (in minutes) (default=15) 
                If err_window > 0:
                    time window = begin_time-err_window to begin_time+(forecast_length+err_window)
                else:
                    time window = begin_time to begin_time + forecast_length

    """
    def __init__(self, reports_path, report_type, initial_time, forecast_length=30, err_window=15):
        
        self.forecast_length = forecast_length
        self.err_window = err_window
        
        if report_type not in ['NOAA', 'IOWA']:
            raise ValueError(f'{report_type} is not valid!')
        
        if len(initial_time) != 12:
                raise ValueError('initial_time format needs to be YYYYMMDDHHmm!')
        
        if report_type == 'IOWA':
            DTYPE = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 'MAG':np.float64, 'TYPETEXT': object}
            COLS = ['VALID', 'LAT', 'LON', 'MAG', 'TYPETEXT']
            self._EVENT_TYPE ='TYPETEXT'
            self._HAIL_CATS = ['HAIL', 'Hail']
            self._WIND_CATS = ['TSTM WND DMG', 'TSTM WIND DMG', 'Tstm Wnd Dmg', 
                             'TSTM WND GST', 'TSTM WND GUST', 'Tstm Wnd Gst']
            self._TORN_CATS = ['TORNADO', 'Tornado']
            
        else:
            DTYPE = {'VALID': np.int64, 'LAT':np.float64, 'LON':np.float64, 
                     'MAG':np.float64, 'EVENT_TYPE':object, 'TOR_F_SCALE' :object}
            
            COLS = ['VALID', 'LAT', 'LON', 'MAG', 'EVENT_TYPE', 'TOR_F_SCALE']
            self._EVENT_TYPE = 'EVENT_TYPE'
            self._HAIL_CATS = 'Hail'
            self._WIND_CATS = 'Thunderstorm Wind'
            self._TORN_CATS = 'Tornado'
            
        # Load the data.
        df = pd.read_csv(reports_path, usecols=COLS, dtype=DTYPE, na_values = 'None')
        df['date'] = pd.to_datetime(df.VALID.astype(str), format='%Y%m%d%H%M')
        self.df = df 
        
        # Get the time window for the reports.
        self.get_time_window(initial_time)

    def __call__(self, ):
        """ Get lats/lons for each hazard type """
        hazards = ['tornado', 'hail', 'wind']
        lsrs = {h : np.array(getattr(self, f'get_{h}_reports')()).T for h in hazards}
                 
        return lsrs
    
    
    def get_points(self, dataset, magnitude='both', hazard = 'all'):
        if magnitude == 'both':
            mag_iterator = ['severe', 'sig_severe']
        else:
            mag_iterator = [magnitude]

        hazard_iterator = ['tornado', 'hail', 'wind'] if hazard=='all' else [hazard]           
        
        points = {}
        
        for magnitude, hazard in itertools.product(mag_iterator, hazard_iterator):
            ll = getattr(self, f'get_{hazard}_reports')(magnitude)
            lsr_x, lsr_y =  self.to_xy(dataset, lats=ll[0], lons=ll[1])
            lsr_points = list(zip(lsr_x, lsr_y))
            
            points[f'{hazard}_{magnitude}'] = lsr_points
        
        return points
        
    
    def to_grid(self, dataset, fname=None, magnitude='both', hazard = 'all', size=3):
        """
        Convert storm reports to a grid. Applies a maximum filter of 3 grid points.
        For a 3 km grid spacing, assumes that reports are potentially valid over a
        9 x 9 km region. 

        Parameters:
        ----------------------------
            dataset, xarray.Dataset

            magnitude, string 

            hazards, string 

        """
        points = self.get_points(dataset, magnitude, hazard)
        data ={}
        
        try:
            nx = len(dataset.NX.values)
        except:
            nx = len(dataset.XLAT.values)
        
        for key in points.keys():
            gridded_reports = self.points_to_grid(xy_pair=points[key], nx=nx)
            data[key] = (['y', 'x'], maximum_filter(gridded_reports, size))
        
        ds = xr.Dataset(data)
        
        if fname is not None:
            print(f'Saving {fname}...')
            ds.to_netcdf(fname)
            ds.close()
        else:
            return ds

            
    def points_to_grid(self, xy_pair, nx):
        """
        Convert points to gridded data
        """
        #xy_pair = [ (x,y) for x,y in xy_pair if x < nx-1 and y < nx-1 and x > 0 and y > 0 ]
        gridded_lsr = np.zeros((nx, nx), dtype=np.int8)
        for i, pair in enumerate(xy_pair):
            gridded_lsr[pair[0],pair[1]] = i+1

        return gridded_lsr


    def get_time_window(self, initial_time): 
        '''
        Get beginning and ending of the time window to search for LSRs
        '''
        # Convert the datetime string to a datetime object 
        start_date = datetime.strptime(initial_time, '%Y%m%d%H%M') 
        end_date = start_date + timedelta(minutes=self.forecast_length+self.err_window)
        
        # Move the start time back (in case reports came in late)
        start_date-= timedelta(minutes=self.err_window)
        
        self.start_date = start_date
        self.end_date = end_date
        
        ##print(f'{self.start_date} - {self.end_date}')
        
        self.time_mask = (self.df.date >= self.start_date) & (self.df.date <= self.end_date)

        #self.time_mask = (self.df.date > self.start_date) & (self.df.date < self.end_date)
        #df = self.df[self.time_mask==True]

        return self

    def _get_event_type(self, etype, names):
        """ Return True/False for whether a row is a given hazard"""
        if not isinstance(names, list):
            names = [names]
            
        return (self.df[etype].isin(names)) 
    
    
    def get_hail_reports(self, magnitude='severe'): 
        '''
        Load hail reports. 

        Parameters:
        ---------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe',  >= 1 in hail size
                if 'sig_severe', >= 2 in hail size

        Returns:
        ---------------------
            lats, lons 

        '''
        magnitude_dict = {'severe' : 1.0,
                           'sig_severe' : 2.0
                          }

        mag_mask = (self.df.MAG >= magnitude_dict[magnitude])
        etype = self._EVENT_TYPE
        event_type_mask = self._get_event_type(etype, self._HAIL_CATS) #(self.df[etype] == self._HAIL_CATS) 
        severe_hail_reports = self.df.loc[self.time_mask & mag_mask & event_type_mask] 

        return ( severe_hail_reports['LAT'].values, severe_hail_reports['LON'].values)

    def get_tornado_reports(self, magnitude='severe'):
        '''
        Load the tornado reports.

        Parameters:
        ----------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe', then use all tornado reports
                if 'sig_severe', >= EF2 tornado reports

        Returns:
        ---------------------
            lats, lons
        '''
        etype = self._EVENT_TYPE 
        event_type_mask = self._get_event_type(etype, self._TORN_CATS)
        
        if magnitude == 'sig_severe': 
            # Tornadoes are not rated in the storm reports database.
            if etype == 'TYPETEXT':
                return ([], [])
            else:
                scales = [ 'EF2', 'EF3', 'EF4', 'EF5']
                mag_mask = self.df.TOR_F_SCALE.isin(scales)            
                total_masks = self.time_mask & event_type_mask & mag_mask
 
        else:
            total_masks = self.time_mask & event_type_mask 

        tornado_reports = self.df.loc[total_masks]
        
        return (tornado_reports['LAT'].values, tornado_reports['LON'].values)       
   
    def get_wind_reports(self, magnitude='severe'):
        '''
        Load the wind reports.

        Parameters:
        ----------------------
            magnitude, 'severe' or 'sig_severe'
                if 'severe', >= 50 kts 
                if 'sig_severe', >= 65 kts

        Returns:
        ---------------------
            lats, lons
        '''
        magnitude_dict = {'severe' : 50.0, 
                          'sig_severe' : 65.0
                         }

        etype = self._EVENT_TYPE 
        event_type_mask = self._get_event_type(etype, self._WIND_CATS)

        dfc = self.df.copy()
        
        # Add a temp magnitude for wind damage reports
        if etype == 'TYPETEXT':
            types = [t for t in self._WIND_CATS if t.split(' ')[-1].upper() == 'DMG']
            inds = self.df['TYPETEXT'].isin(types)==True
            dfc.loc[inds, 'MAG'] = 55.0
        
        mag_mask = (dfc.MAG >= magnitude_dict[magnitude])
        wind_reports = dfc.loc[self.time_mask & mag_mask & event_type_mask]

        return (wind_reports['LAT'].values, wind_reports['LON'].values) 

    def to_xy(self, ds, lats, lons):
        """Uses a KD-tree approach to determine, which i,j index an
        lat/lon coordiante pair is closest to. Used to map storm reports to 
        the WoFS domain"""
        
        try:
            wofs_lat = np.round(ds.xlat.values,7)
            wofs_lon = np.round(ds.xlon.values,7)
        except:
            wofs_lat = np.round(ds.XLAT.values,7)
            wofs_lon = np.round(ds.XLON.values,7)
        
        max_points = wofs_lat.shape[0]*wofs_lat.shape[1]
        
        min_lat, max_lat = np.min(wofs_lat), np.max(wofs_lat)
        min_lon, max_lon = np.min(wofs_lon), np.max(wofs_lon)

        lons_vect = [min_lon, max_lon, max_lon, min_lon ]
        lats_vect = [min_lat, min_lat, max_lat, max_lat]

        lons_lats_vect = np.column_stack((lons_vect, lats_vect)) # Reshape coordinates
        polygon = Polygon(lons_lats_vect) # create polygon
        
        good_pairs = [(lon, lat) for lon, lat in zip(lons, lats) if polygon.contains(Point(lon, lat))]
        lons_within = [pair[0] for pair in good_pairs]
        lats_within = [pair[1] for pair in good_pairs]
        
        grid = pyresample.geometry.GridDefinition(lats=wofs_lat, lons=wofs_lon)
        swath = pyresample.geometry.SwathDefinition(lons=lons_within, lats=lats_within)

        # Determine nearest (w.r.t. great circle distance) neighbour in the grid.
        _, _, index_array, distance_array = pyresample.kd_tree.get_neighbour_info(
            source_geo_def=grid, target_geo_def=swath, radius_of_influence=50000,
            neighbours=1)
        
        index_array[index_array==max_points]=max_points-1
        
        ###print(f'{index_array=}')
        
        # get_neighbour_info() returns indices in the flattened lat/lon grid. Compute
        # the 2D grid indices:
        x,y = np.unravel_index(index_array, grid.shape)

        return x,y 

